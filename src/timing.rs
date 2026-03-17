//! Precise wall-clock timing for pipeline stage comparison.
//!
//! ## Design goals
//!
//! - **No external deps** — built entirely on `std::time::Instant`.
//! - **Sub-task granularity** — each significant computational step (not just
//!   the 5 numbered stages) gets its own entry, so methods can be compared
//!   fairly.  For example `similarity::dense` and `similarity::knn` are separate
//!   entries even though both live inside stage 3.
//! - **Execution order preserved** — entries are stored in a `Vec` in the order
//!   they were recorded, so the printed table reads like a timeline.
//! - **Two outputs**:
//!   1. A formatted table to stderr during the run (human-readable).
//!   2. A `timing.json` file in the output directory (machine-readable,
//!      load into Python/pandas for cross-run comparison plots).
//! - **Params captured per entry** — every entry stores a short string
//!   describing the configuration (e.g. `"knn k=20 N=4096"`), so two JSON
//!   files from different runs can be diffed unambiguously.
//!
//! ## Usage pattern
//!
//! ```rust
//! let mut t = Timer::new();
//!
//! let guard = t.start("similarity::knn", "k=20 N=4096");
//! let sim = build_knn(...);
//! t.stop(guard);   // duration recorded
//!
//! // or use the closure form:
//! let sim = t.measure("similarity::knn", "k=20 N=4096", || build_knn(...));
//!
//! t.print_table();
//! t.write_json(&output_dir)?;
//! ```

use std::time::{Duration, Instant};
use std::path::Path;

// ── Core types ────────────────────────────────────────────────────────────────

/// A single completed timing measurement.
#[derive(Debug, Clone)]
pub struct TimingEntry {
    /// Short identifier, e.g. `"similarity::knn"` or `"eigensolver::lanczos"`.
    pub name: String,
    /// Human-readable parameter summary, e.g. `"k=20 N=4096 σ=0.1"`.
    pub params: String,
    /// Wall-clock duration.
    pub duration: Duration,
}

impl TimingEntry {
    /// Duration as milliseconds (f64, sub-ms precision).
    pub fn ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }
}

/// An in-progress timing guard returned by [`Timer::start`].
/// Drop it (or pass to [`Timer::stop`]) to record the duration.
pub struct Guard {
    name: String,
    params: String,
    start: Instant,
}

/// Collects named timing entries in execution order.
pub struct Timer {
    entries: Vec<TimingEntry>,
    /// Wall-clock instant when the Timer was created (= program start).
    wall_start: Instant,
}

impl Timer {
    /// Create a new timer, recording the current instant as t=0.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            wall_start: Instant::now(),
        }
    }

    /// Start timing a named task.  Returns a [`Guard`]; pass it to
    /// [`stop`](Timer::stop) when the task is done.
    pub fn start(&self, name: impl Into<String>, params: impl Into<String>) -> Guard {
        Guard {
            name: name.into(),
            params: params.into(),
            start: Instant::now(),
        }
    }

    /// Stop a guard and record the elapsed duration.
    pub fn stop(&mut self, guard: Guard) -> Duration {
        let duration = guard.start.elapsed();
        self.entries.push(TimingEntry {
            name: guard.name,
            params: guard.params,
            duration,
        });
        duration
    }

    /// Time a closure and record the result.  Returns the closure's value.
    pub fn measure<T>(
        &mut self,
        name: impl Into<String>,
        params: impl Into<String>,
        f: impl FnOnce() -> T,
    ) -> T {
        let g = self.start(name, params);
        let result = f();
        self.stop(g);
        result
    }

    /// All recorded entries in execution order.
    pub fn entries(&self) -> &[TimingEntry] {
        &self.entries
    }

    /// Total wall-clock time since the timer was created.
    pub fn total_elapsed(&self) -> Duration {
        self.wall_start.elapsed()
    }

    // ── Human-readable output ─────────────────────────────────────────────────

    /// Print a formatted timing table to stderr.
    ///
    /// Example output:
    /// ```text
    /// ┌─────────────────────────────────┬────────────────────────────┬──────────────┬────────┐
    /// │ task                            │ params                     │     time(ms) │  cumul │
    /// ├─────────────────────────────────┼────────────────────────────┼──────────────┼────────┤
    /// │ features                        │ N=4096                     │        2.31  │   2.3% │
    /// │ similarity::knn                 │ k=20 N=4096 σc=0.10        │      187.44  │  92.1% │
    /// │ laplacian                       │ N=4096                     │        8.22  │  96.1% │
    /// │ eigensolver::lanczos            │ k=7 q=41 N=4096            │       14.77  │  99.6% │
    /// │ render                          │ 7 images                   │        0.83  │ 100.0% │
    /// ├─────────────────────────────────┼────────────────────────────┼──────────────┼────────┤
    /// │ TOTAL                           │                            │      213.57  │        │
    /// └─────────────────────────────────┴────────────────────────────┴──────────────┴────────┘
    /// ```
    pub fn print_table(&self) {
        if self.entries.is_empty() {
            eprintln!("[timing] no entries recorded");
            return;
        }

        let total_ms: f64 = self.entries.iter().map(|e| e.ms()).sum();
        let total_ms = total_ms.max(1e-6); // avoid div-by-zero

        // Column widths (minimum widths, expand to fit content)
        let name_w  = self.entries.iter().map(|e| e.name.len()).max().unwrap_or(4)
                          .max(4).min(40);
        let param_w = self.entries.iter().map(|e| e.params.len()).max().unwrap_or(6)
                          .max(6).min(36);
        // time and cumul columns are fixed width
        let time_w  = 12usize; // "  12345.678 "
        let pct_w   =  7usize; // " 100.0% "

        let row_width = 3 + name_w + 3 + param_w + 3 + time_w + 3 + pct_w + 1;
        let rule_top    = format!("┌{}┬{}┬{}┬{}┐",
            "─".repeat(name_w  + 2), "─".repeat(param_w + 2),
            "─".repeat(time_w  + 2), "─".repeat(pct_w   + 2));
        let rule_mid    = format!("├{}┼{}┼{}┼{}┤",
            "─".repeat(name_w  + 2), "─".repeat(param_w + 2),
            "─".repeat(time_w  + 2), "─".repeat(pct_w   + 2));
        let rule_bot    = format!("└{}┴{}┴{}┴{}┘",
            "─".repeat(name_w  + 2), "─".repeat(param_w + 2),
            "─".repeat(time_w  + 2), "─".repeat(pct_w   + 2));

        let header = format!("│ {:<name_w$} │ {:<param_w$} │ {:>time_w$} │ {:>pct_w$} │",
            "task", "params", "time (ms)", "% total");

        eprintln!("\n{}", rule_top);
        eprintln!("{}", header);
        eprintln!("{}", rule_mid);

        let mut cumul_ms = 0.0f64;
        for e in &self.entries {
            cumul_ms += e.ms();
            let pct = cumul_ms / total_ms * 100.0;
            let name_trunc  = truncate(&e.name,   name_w);
            let param_trunc = truncate(&e.params, param_w);
            eprintln!("│ {:<name_w$} │ {:<param_w$} │ {:>time_w$.3} │ {:>6.1}% │",
                name_trunc, param_trunc, e.ms(), pct);
        }

        eprintln!("{}", rule_mid);
        let total_label = "TOTAL";
        let total_wall  = self.total_elapsed().as_secs_f64() * 1000.0;
        eprintln!("│ {:<name_w$} │ {:<param_w$} │ {:>time_w$.3} │ {:>pct_w$} │",
            total_label, format!("wall {:.1} ms", total_wall), total_ms, "");
        eprintln!("{}", rule_bot);
        let _ = row_width; // suppress unused warning
    }

    // ── JSON output ───────────────────────────────────────────────────────────

    /// Write timing data as a JSON file to `<output_dir>/timing.json`.
    ///
    /// Schema:
    /// ```json
    /// {
    ///   "total_wall_ms": 213.57,
    ///   "entries": [
    ///     { "name": "features", "params": "N=4096", "ms": 2.31 },
    ///     ...
    ///   ]
    /// }
    /// ```
    pub fn write_json(&self, output_dir: &Path) -> anyhow::Result<()> {
        use std::fmt::Write;

        let total_wall_ms = self.total_elapsed().as_secs_f64() * 1000.0;

        let mut s = String::new();
        writeln!(s, "{{")?;
        writeln!(s, "  \"total_wall_ms\": {:.6},", total_wall_ms)?;
        writeln!(s, "  \"entries\": [")?;

        for (i, e) in self.entries.iter().enumerate() {
            let comma = if i + 1 < self.entries.len() { "," } else { "" };
            // Escape any double-quotes in name/params (shouldn't happen, but be safe)
            let name_esc   = e.name.replace('"', "\\\"");
            let params_esc = e.params.replace('"', "\\\"");
            writeln!(
                s,
                "    {{ \"name\": \"{}\", \"params\": \"{}\", \"ms\": {:.6} }}{}",
                name_esc, params_esc, e.ms(), comma
            )?;
        }

        writeln!(s, "  ]")?;
        writeln!(s, "}}")?;

        let path = output_dir.join("timing.json");
        std::fs::write(&path, &s)?;
        eprintln!("      timing.json  ({} entries)", self.entries.len());
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Truncate a string to at most `max_chars` characters, appending "…" if cut.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_owned()
    } else if max_chars > 1 {
        format!("{}…", &s[..max_chars - 1])
    } else {
        s[..max_chars].to_owned()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn records_entries_in_order() {
        let mut t = Timer::new();
        t.measure("a", "p1", || thread::sleep(Duration::from_millis(1)));
        t.measure("b", "p2", || thread::sleep(Duration::from_millis(1)));
        t.measure("c", "p3", || {});
        assert_eq!(t.entries().len(), 3);
        assert_eq!(t.entries()[0].name, "a");
        assert_eq!(t.entries()[1].name, "b");
        assert_eq!(t.entries()[2].name, "c");
    }

    #[test]
    fn guard_start_stop() {
        let mut t = Timer::new();
        let g = t.start("task", "params");
        thread::sleep(Duration::from_millis(2));
        let d = t.stop(g);
        assert!(d >= Duration::from_millis(1), "expected ≥1ms, got {:?}", d);
        assert_eq!(t.entries().len(), 1);
        assert!(t.entries()[0].ms() >= 1.0);
    }

    #[test]
    fn measure_returns_value() {
        let mut t = Timer::new();
        let x: u32 = t.measure("compute", "", || 42);
        assert_eq!(x, 42);
    }

    #[test]
    fn duration_is_positive() {
        let mut t = Timer::new();
        t.measure("noop", "", || {});
        assert!(t.entries()[0].duration >= Duration::ZERO);
    }

    #[test]
    fn total_elapsed_grows() {
        let t = Timer::new();
        thread::sleep(Duration::from_millis(5));
        assert!(t.total_elapsed() >= Duration::from_millis(4));
    }

    #[test]
    fn write_json_produces_valid_content() {
        let mut t = Timer::new();
        t.measure("stage1", "N=100", || thread::sleep(Duration::from_millis(1)));
        t.measure("stage2", "k=5",  || {});

        let dir = std::env::temp_dir();
        t.write_json(&dir).expect("write_json failed");

        let content = std::fs::read_to_string(dir.join("timing.json"))
            .expect("could not read timing.json");
        assert!(content.contains("\"entries\""), "missing entries key");
        assert!(content.contains("\"stage1\""));
        assert!(content.contains("\"stage2\""));
        assert!(content.contains("\"total_wall_ms\""));
        // Should be valid enough to not have unmatched braces
        assert_eq!(
            content.chars().filter(|&c| c == '{').count(),
            content.chars().filter(|&c| c == '}').count()
        );
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 5),  "hello");
    }

    #[test]
    fn truncate_long_string_appends_ellipsis() {
        let s = truncate("similarity::knn", 10);
        // The result must fit in 10 chars (char count, not byte count — '…' is 3 bytes)
        assert!(s.chars().count() <= 10, "got {} chars: {:?}", s.chars().count(), s);
        assert!(s.ends_with('…'), "expected ellipsis at end of {:?}", s);
    }

    #[test]
    fn empty_timer_prints_without_panic() {
        let t = Timer::new();
        t.print_table(); // should not panic
    }
}
