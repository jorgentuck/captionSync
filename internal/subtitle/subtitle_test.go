package subtitle

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, name string, data []byte) string {
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func plus(d float64) func(s, e float64) (float64, float64) {
	return func(s, e float64) (float64, float64) { return s + d, e + d }
}

func TestSRTOnlyTimesChange(t *testing.T) {
	src := "\uFEFF1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>Hello</i> there\r\n\r\n2\r\n00:01:00,250 --> 00:01:03,000 X1:10\r\nSecond line\r\nwith two rows\r\n\r\n"
	f, err := Load(write(t, "a.srt", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	cues := f.Cues()
	if len(cues) != 2 || cues[1].Text != "Second line\nwith two rows" || cues[0].Start != 1 {
		t.Fatalf("bad parse: %+v", cues)
	}
	if n := f.Retime(plus(1.5)); n != 2 {
		t.Fatalf("changed %d", n)
	}
	want := strings.NewReplacer("00:00:01,000 --> 00:00:02,500", "00:00:02,500 --> 00:00:04,000",
		"00:01:00,250 --> 00:01:03,000", "00:01:01,750 --> 00:01:04,500").Replace(src)
	if got := string(f.Bytes()); got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestVTT(t *testing.T) {
	src := "WEBVTT\n\nNOTE hi\n\n00:05.000 --> 00:07.000 align:start\nShort hours\n\nid2\n01:02:03.004 --> 01:02:04.000\nFull\n"
	f, err := Load(write(t, "a.vtt", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	f.Retime(plus(-1))
	got := string(f.Bytes())
	if !strings.Contains(got, "00:00:04.000 --> 00:00:06.000 align:start\nShort hours") ||
		!strings.Contains(got, "01:02:02.004 --> 01:02:03.000\nFull\n") || !strings.HasPrefix(got, "WEBVTT\n\nNOTE hi") {
		t.Fatalf("got %q", got)
	}
}

func TestASS(t *testing.T) {
	src := "[Script Info]\nTitle: x\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" +
		"Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,{\\i1}Hello, world{\\i0}\n" +
		"Comment: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,note\n"
	f, err := Load(write(t, "a.ass", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	if c := f.Cues(); len(c) != 1 || c[0].Text != "{\\i1}Hello, world{\\i0}" {
		t.Fatalf("cues %+v", c)
	}
	f.Retime(plus(0.25))
	got := string(f.Bytes())
	if !strings.Contains(got, "Dialogue: 0,0:00:01.25,0:00:02.75,Default,,0,0,0,,{\\i1}Hello, world{\\i0}") ||
		!strings.Contains(got, "Comment: 0,0:00:03.25,0:00:04.25,") {
		t.Fatalf("got %q", got)
	}
}

func TestLegacyEncodingPreserved(t *testing.T) {
	src := []byte("1\n00:00:01,000 --> 00:00:02,000\nCaf\xe9 \x93quoted\x94\n")
	f, err := Load(write(t, "a.srt", src))
	if err != nil {
		t.Fatal(err)
	}
	if f.Cues()[0].Text != "Café “quoted”" {
		t.Fatalf("text %q", f.Cues()[0].Text)
	}
	f.Retime(plus(1))
	if got := f.Bytes(); string(got) != strings.Replace(string(src), "00:00:01,000 --> 00:00:02,000", "00:00:02,000 --> 00:00:03,000", 1) {
		t.Fatalf("bytes changed: %q", got)
	}
}

func TestNegativeClamped(t *testing.T) {
	f, _ := Load(write(t, "a.srt", []byte("1\n00:00:01,000 --> 00:00:02,000\nx\n")))
	f.Retime(plus(-5))
	if !strings.Contains(string(f.Bytes()), "00:00:00,000 --> 00:00:00,010") {
		t.Fatalf("got %q", f.Bytes())
	}
}

func identity(s, e float64) (float64, float64) { return s, e }

// Round trip with no shift must reproduce the file byte for byte, whatever its quirks.
func TestIdentityRoundTrip(t *testing.T) {
	cases := map[string]string{
		"mixed.srt":   "1\r\n00:00:01,000 --> 00:00:02,000\r\nCRLF line\n\n2\n00:00:03,000 --> 00:00:04,000\nLF line\n",
		"nofinal.srt": "1\n00:00:01,000 --> 00:00:02,000\nno final newline",
		"bomtime.srt": "\uFEFF00:00:01,000 --> 00:00:02,000\nfirst cue, no index\n\n00:00:03,000 --> 00:00:04,000\nsecond\n",
		"crlf.ass":    "[Events]\r\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hi, there\r\n",
	}
	for name, src := range cases {
		f, err := Load(write(t, name, []byte(src)))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		f.Retime(identity)
		if got := string(f.Bytes()); got != src {
			t.Errorf("%s: identity round trip changed bytes:\n got %q\nwant %q", name, got, src)
		}
	}
}

func TestBOMBeforeFirstTimingLineIsRetimed(t *testing.T) {
	src := "\uFEFF00:00:01,000 --> 00:00:02,000\nfirst\n\n00:00:03,000 --> 00:00:04,000\nsecond\n"
	f, err := Load(write(t, "b.srt", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Cues()) != 2 {
		t.Fatalf("cues %d, want 2 (BOM hid the first)", len(f.Cues()))
	}
	f.Retime(plus(1))
	if got := string(f.Bytes()); !strings.HasPrefix(got, "\uFEFF00:00:02,000 --> 00:00:03,000\n") {
		t.Fatalf("got %q", got)
	}
}

func TestMixedLineEndingsPreservedWhenShifting(t *testing.T) {
	src := "1\r\n00:00:01,000 --> 00:00:02,000\r\nA\r\n\r\n2\n00:00:03,000 --> 00:00:04,000\nB\n"
	f, _ := Load(write(t, "m.srt", []byte(src)))
	f.Retime(plus(1))
	want := "1\r\n00:00:02,000 --> 00:00:03,000\r\nA\r\n\r\n2\n00:00:04,000 --> 00:00:05,000\nB\n"
	if got := string(f.Bytes()); got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestUnpaddedMillisecondsAreMilliseconds(t *testing.T) {
	f, _ := Load(write(t, "u.srt", []byte("1\n00:00:01,5 --> 00:00:02,50\nx\n")))
	c := f.Cues()[0]
	if c.Start != 1.005 || c.End != 2.05 {
		t.Fatalf("got %v %v, want 1.005 2.05 (as ffmpeg/VLC read it)", c.Start, c.End)
	}
}

func TestShortASSLineStillRetimed(t *testing.T) {
	src := "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" +
		"Dialogue: 0,0:00:01.00,0:00:02.00,Default\n" + // malformed: too few fields
		"Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,ok\n"
	f, err := Load(write(t, "s.ass", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	f.Retime(plus(1))
	got := string(f.Bytes())
	if !strings.Contains(got, "Dialogue: 0,0:00:02.00,0:00:03.00,Default\n") {
		t.Fatalf("short line not retimed: %q", got)
	}
}

func TestASSRoundTwo(t *testing.T) {
	// CRLF short line keeps its \r; Text not last; Sound events shift; trailing-comma Format; integer cs
	src := "[Events]\r\n" +
		"Format: Layer, Start, End, Text, Style\r\n" +
		"Dialogue: 0,0:00:01.00,0:00:02.00,Hello there friend,Default\r\n" +
		"Sound: 0,0:00:03.00,0:00:04.00,boom.wav,Default\r\n" +
		"Dialogue: 0,0:00:05.5,0:00:06.00\r\n"
	f, err := Load(write(t, "r2.ass", []byte(src)))
	if err != nil {
		t.Fatal(err)
	}
	c := f.Cues()
	if len(c) != 1 || c[0].Text != "Hello there friend" {
		t.Fatalf("cues %+v", c)
	}
	f.Retime(identity)
	if got := string(f.Bytes()); got != strings.Replace(src, "0:00:05.5,", "0:00:05.05,", 1) {
		t.Fatalf("identity changed more than the normalised time:\n%q", got)
	}
	f.Retime(plus(1))
	got := string(f.Bytes())
	for _, want := range []string{"Sound: 0,0:00:04.00,0:00:05.00,boom.wav,Default\r\n", "Dialogue: 0,0:00:06.05,0:00:07.00\r\n"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %q", want, got)
		}
	}
	g, _ := Load(write(t, "tc.ass", []byte("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text,\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,no commas here\n")))
	if c := g.Cues(); len(c) != 1 || c[0].Text != "no commas here" {
		t.Fatalf("trailing-comma format: %+v", c)
	}
}
