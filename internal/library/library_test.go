package library

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jorgentuck/captionSync/internal/config"
)

func lib(t *testing.T, roots ...string) *Library {
	cfg := &config.Config{MediaRoots: roots, VideoExt: []string{".mkv", ".mp4"}, SubExt: []string{".srt"},
		IgnorePatterns: []string{".*", "*.captionsync.tmp*"}}
	return New(cfg)
}

func touch(t *testing.T, p string) {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestSameStemVideosDoNotShareSubtitles(t *testing.T) {
	d := t.TempDir()
	touch(t, filepath.Join(d, "Movie.mkv"))
	touch(t, filepath.Join(d, "Movie.mp4"))
	touch(t, filepath.Join(d, "Movie.en.srt"))
	l := lib(t, d)
	if got := l.Sidecars(filepath.Join(d, "Movie.mkv")); len(got) != 1 {
		t.Fatalf("mkv should own the subtitle, got %v", got)
	}
	if got := l.Sidecars(filepath.Join(d, "Movie.mp4")); len(got) != 0 {
		t.Fatalf("mp4 must not also claim it, got %v", got)
	}
	if v := l.VideoForSubtitle(filepath.Join(d, "Movie.en.srt")); filepath.Base(v) != "Movie.mkv" {
		t.Fatalf("subtitle resolved to %s", v)
	}
}

func TestSymlinkCannotEscapeRoot(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	touch(t, filepath.Join(outside, "secret.mkv"))
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Skip("symlinks unsupported")
	}
	l := lib(t, root)
	if l.Allowed(filepath.Join(root, "link", "secret.mkv")) {
		t.Fatal("symlink led outside MEDIA_ROOTS")
	}
	if !l.Allowed(filepath.Join(root, "real.mkv")) {
		t.Fatal("normal path inside root rejected")
	}
	if l.Allowed(root + "2/x.mkv") {
		t.Fatal("prefix confusion: /media2 accepted for /media")
	}
}
