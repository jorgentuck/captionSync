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
