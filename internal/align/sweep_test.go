package align

import (
	"os"
	"testing"
)

// Many-seed robustness sweep. Run with SWEEP=1 go test -run Sweep -v ./internal/align
func TestSweep(t *testing.T) {
	if os.Getenv("SWEEP") == "" {
		t.Skip("set SWEEP=1")
	}
	type tc struct {
		name  string
		mk    func(seed int64) *scene
		tol   float64
		check func(r Result) bool
	}
	cases := []tc{
		{"drift", func(sd int64) *scene {
			return &scene{cues: 1100, yield: 0.13, noise: 0.2, outliers: 0.15, seed: sd,
				subOf: func(a float64) float64 { return a + 1.7 + a*3.3/3600 }}
		}, 0.25, func(r Result) bool { return r.Status == "needs_fix" && len(r.Mapping.Pieces) == 1 }},
		{"constant", func(sd int64) *scene {
			return &scene{cues: 390, yield: 0.18, noise: 0.22, outliers: 0.2, seed: sd,
				subOf: func(a float64) float64 { return a - 0.6 }}
		}, 0.2, func(r Result) bool { return r.Status == "needs_fix" }},
		{"jump", func(sd int64) *scene {
			return &scene{cues: 900, yield: 0.15, noise: 0.2, outliers: 0.15, seed: sd,
				subOf: func(a float64) float64 {
					if a > 2400 {
						return a + 3.2
					}
					return a + 0.9
				}}
		}, 0.25, func(r Result) bool { return r.Status == "needs_fix" && len(r.Mapping.Pieces) == 2 }},
		{"in-sync", func(sd int64) *scene {
			return &scene{cues: 600, yield: 0.15, noise: 0.22, outliers: 0.2, seed: sd,
				subOf: func(a float64) float64 { return a + 0.05 }}
		}, 0.2, func(r Result) bool { return r.Status == "in_sync" }},
	}
	for _, c := range cases {
		bad := 0
		for sd := int64(100); sd < 140; sd++ {
			sc := c.mk(sd)
			r := run(sc, false)
			if !c.check(r) || (r.Status == "needs_fix" && accuracy(sc, r, c.tol) < 0.95) {
				bad++
				t.Logf("%s seed %d: %s %s", c.name, sd, r.Status, r.Message)
			}
		}
		t.Logf("%-9s %d/40 bad", c.name, bad)
	}
}
