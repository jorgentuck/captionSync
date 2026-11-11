package align

import (
	"encoding/json"
	"math"
	"strconv"
)

func itoa(n int64) string { return strconv.FormatInt(n, 10) }

func signed(v float64) string {
	n := int64(math.Round(v))
	if n >= 0 {
		return "+" + itoa(n)
	}
	return itoa(n)
}

func signedF(v float64) string {
	s := strconv.FormatFloat(v, 'f', 2, 64)
	if v >= 0 {
		return "+" + s
	}
	return s
}

// MarshalJSON writes infinite piece bounds as null.
func (p Piece) MarshalJSON() ([]byte, error) {
	f := func(v float64) any {
		if math.IsInf(v, 0) {
			return nil
		}
		return math.Round(v*1000) / 1000
	}
	return json.Marshal(map[string]any{
		"start": f(p.Start), "end": f(p.End), "offset_ms": math.Round(p.Offset * 1000),
		"drift_s_per_hour": math.Round(p.Slope*3600*100) / 100, "anchors": p.Anchors,
	})
}
