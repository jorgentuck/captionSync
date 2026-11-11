# captionSync

Fixes **subtitle timing** on your videos. Use it when subtitles from Bazarr show up too early or too late.

It does **not** generate captions. For each video, a small speech model (whisper.cpp) listens to parts of the audio to find *when* dialogue is spoken. CaptionSync then matches your existing subtitle lines to those moments and rewrites **only the timestamps** in the `.srt` / `.ass` / `.vtt` sidecar files. The subtitle text, styling, encoding and line endings stay byte-for-byte the same. Video files are only read, unless you opt in to fixing embedded subtitles with `EMBEDDED_SUBS=remux`.

It handles:

| Problem | How it's fixed |
|---|---|
| Whole file early/late by N seconds | Single offset |
| Subs made for a different framerate (drift grows over the episode) | Detects the 23.976/24/25/29.97 ratio, then offsets |
| Different cut / ad breaks (offset jumps partway through) | Splits into segments, each with its own offset |
| Subtitle language ≠ audio language | Falls back to matching "speech happening" against "subtitle on screen" (global offset only, flagged low confidence) |

### Embedded subtitles

Some videos carry subtitles inside the file (e.g. an English track muxed into the `.mkv`). The `EMBEDDED_SUBS` setting decides what happens to them:

| `EMBEDDED_SUBS` | What happens |
|---|---|
| `report` *(default)* | Detected and listed, never touched. A video whose **only** subtitles are embedded gets the status **Embedded only**, so you can filter for them to see how many you have. |
| `remux` | Embedded text tracks are checked. Fixed tracks are written back by rewriting the video container: every other stream is copied untouched, language/title/default/forced flags are kept, and the new file is verified (stream count, duration) before it replaces the original. Needs free space for one temporary copy next to the video. |
| `sidecar` | Fixed embedded tracks are written as a new file next to the video (e.g. `Movie.eng.srt`). The video is left untouched. Your player will show both, so pick the external one. |
| `off` | Embedded subtitles are ignored. |

To switch later: change the setting, restart the container, then click **Re-check embedded-only**.

Image-based subtitles (PGS from Blu-ray, VobSub from DVD) are bitmaps, not text, so they can't be matched to speech. They're listed but never changed.

With `remux`, the video file changes, so Tdarr may notice it and run it through its flow again. If that happens, add a Tdarr filter or use `sidecar`.

Every video is recorded in a SQLite database together with a signature of the video and its subtitle files. Nothing gets processed twice unless something changed or you force it.

## Performance (1000+ videos, 8 GB GPU)

* **Go + whisper.cpp**: one static Go binary drives a C++/CUDA whisper server. No Python.
* **Sampled listening**: by default it transcribes only 12 × 60 s windows spread across each video, not the whole file. If the result is unclear, or the offset changes partway through, it automatically does a full pass for that file only.
* **Small VRAM footprint**: the default `small-q5_1` model uses well under 1 GB of VRAM. The whisper server **shuts down after 5 idle minutes** (`MODEL_IDLE_UNLOAD`), so the GPU is fully released for Tdarr and Plex.
* **Overlap**: ffmpeg decodes the next audio window while the GPU transcribes the current one, and `WORKERS` files run in parallel.

Expect roughly 10–30 s per video on a mid-range GPU in sample mode.

## Install (Unraid)

The image is built automatically by GitHub Actions and published as
**`ghcr.io/jorgentuck/captionsync:latest`**. Unraid only has to pull it.

1. **GPU:** install the **Nvidia Driver** plugin (Apps tab) and reboot if it asks.
   Copy your GPU's UUID from *Settings → Nvidia Driver*, or just use `all`.
2. **Add the template.** In the Unraid web terminal (top-right `>_` icon), run:
   ```bash
   wget -O /boot/config/plugins/dockerMan/templates-user/my-captionSync.xml \
     https://raw.githubusercontent.com/jorgentuck/captionSync/main/unraid/captionsync.xml
   ```
3. Go to **Docker → Add Container**. Under **Template**, pick **captionSync**
   (in the *User templates* section).
4. Check the settings:
   * **Movies** and **TV**: your two library folders, e.g. `/mnt/user/movies` and `/mnt/user/tv`. Use the folder picker to choose your real paths.
     Inside the container they appear as `/media/movies` and `/media/tv`, and both are scanned and watched. They must stay **read-write**, because subtitle files are rewritten.
     For a third folder (e.g. anime), click *Add another Path* and set its container path to `/media/anime`.
   * **AppData**: `/mnt/user/appdata/captionsync`
   * **NVIDIA_VISIBLE_DEVICES**: your GPU UUID or `all`. The template already adds `--runtime=nvidia`.
   * **Dry run**: set `true` for your first pass if you'd like to see what it would change first.
5. Click **Apply**, then open the WebUI (port 8080).

The first run downloads the whisper model (~190 MB) and a tiny voice-activity model into `/config/models`.

**Paths for hooks.** captionSync sees your folders as `/media/movies/...` and `/media/tv/...`. The Bazarr and
Tdarr hooks send paths the way *those* containers see them. Look at their path mappings and set **Path mappings**
to match. For example, if Bazarr has `/mnt/user/movies → /movies` and `/mnt/user/tv → /tv`, use
`/movies:/media/movies,/tv:/media/tv`. The file watcher doesn't need this.

**Updating:** after a new image is published, click the container's icon → **Check for Updates** (or
**Force Update**).

**Tip:** start with `DRY_RUN=true` and click **Process all**. You'll see what *would* change (status
"Needs fix") without touching anything. Then set it back to `false` and click **Re-run "needs fix"**.

Plain Docker / Compose: see `docker-compose.yml`. To build the image yourself: `docker build -t captionsync .`
(the first build compiles whisper.cpp with CUDA and takes a while).

## Processing options (web UI)

* **Process path**: a single file, or a folder (recursive). Pick one with the browser or paste a path.
* **Process all new / changed**: walks every media root and queues videos that are new or whose video/subtitle files changed since they were last handled.
* **Re-check "in sync"**: re-processes the files previously judged to need no change.
* **Retry uncertain / failed**, and **Re-run "needs fix"** (after a dry run).
* **Force**: re-check even if nothing changed.
* **Pause / Clear queue**.

**Queue order:** new arrivals (watcher, Bazarr/Tdarr hooks) and single files you pick go first ("next").
Folders and status re-checks you start come after them ("picked"), and **Process all** / scheduled scans run in
the background ("bulk"). So a multi-day first pass never delays new subtitles. If a file waiting in the bulk
queue gets a hook or watcher event, it moves up.

Click any file for per-subtitle details: matched lines, shift per segment, drift, and history.

## Automatic triggers

### File watcher (Tdarr, Bazarr, anything)
CaptionSync watches `/media` (inotify). When a video or subtitle file appears or changes, it waits until the file has stopped changing for `SETTLE_SECONDS` (default 90). This covers Tdarr copying the transcoded file back in. Then it queues the video.

Files whose signature hasn't changed are skipped, so CaptionSync's own rewrites don't loop. Tdarr cache/temp files are ignored (`IGNORE_PATTERNS`).

If inotify doesn't fire on your setup (some network mounts), use `WATCH_MODE=poll`, or schedule a cheap periodic sweep with `SCAN_INTERVAL_MINUTES=60`.

### Bazarr post-processing hook (recommended)
In Bazarr, go to **Settings → Subtitles → Custom Post-Processing**, enable it, and set the command to:
```bash
curl -s -X POST "http://UNRAID-IP:8080/api/hooks/bazarr" --data-urlencode "path={{episode}}"
```
If Bazarr sees the library under a different path (e.g. `/tv`), add `PATH_MAPPINGS=/tv:/media/tv,/movies:/media/movies`.

If you use this, turn **off** Bazarr's own *automatic subtitle synchronization*, so the two don't both adjust the same file.

### Tdarr / anything else
`POST /api/hooks/<name>` with `path=<file>` as a query parameter, form field or JSON (`{"path": "..."}`). Subtitle paths work too: they're resolved to their video.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MEDIA_ROOTS` | `/media` | Comma-separated library roots (only paths inside these are touched) |
| `WHISPER_MODEL` | `small-q5_1` | ggml model name from [whisper.cpp models](https://huggingface.co/ggerganov/whisper.cpp) or a path to a `.bin`. `base-q5_1` is lighter; `small.en-q5_1` is better for English-only libraries |
| `WHISPER_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `WHISPER_VAD` | `true` | Silero voice-activity detection (885 KB model, auto-downloaded). Makes speech start times accurate to ~30 ms; leave on |
| `MODEL_IDLE_UNLOAD` | `300` | Seconds idle before the whisper server is stopped and VRAM is freed |
| `AUDIO_LANGUAGE` | *(auto)* | Force audio language, e.g. `en` |
| `PREFERRED_AUDIO_LANG` | | e.g. `eng,en`: which audio track to listen to when there are several |
| `TRANSCRIBER` | `local` | `openai` = any OpenAI-compatible endpoint (OpenAI, Groq, self-hosted) |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI / – / `whisper-1` | For `TRANSCRIBER=openai` |
| `WORKERS` | `2` | Files processed in parallel |
| `ANALYSIS_MODE` | `sample` | `sample` or `full` |
| `SAMPLE_WINDOWS` / `SAMPLE_WINDOW_SECONDS` | `12` / `60` | Sampled listening |
| `AUTO_FULL` | `true` | Escalate to a full pass when sampling is inconclusive |
| `SYNC_THRESHOLD_MS` | `300` | Deviations below this count as "in sync" |
| `MAX_OFFSET_SECONDS` | `120` | Largest offset searched for |
| `MIN_ANCHORS` | `8` | Matched lines needed for a confident fix |
| `MATCH_THRESHOLD` | `80` | Fuzzy text match score (0–100) |
| `ALLOW_ACTIVITY_FALLBACK` | `true` | Language-agnostic fallback |
| `EMBEDDED_SUBS` | `report` | `report`, `remux`, `sidecar`, `off` (see *Embedded subtitles*) |
| `DRY_RUN` | `false` | Analyse and report, never write |
| `BACKUP` | `true` | Keep originals in `/config/backups/<date>/…` |
| `WATCH_MODE` | `inotify` | `inotify`, `poll`, `off` |
| `WATCH_PATHS` | `MEDIA_ROOTS` | Folders to watch |
| `SETTLE_SECONDS` | `90` | Quiet period before a changed file is processed |
| `POLL_INTERVAL` | `60` | Seconds between polls (`WATCH_MODE=poll`) |
| `SCAN_INTERVAL_MINUTES` | `0` | Periodic "process all new/changed" sweep |
| `SCAN_ON_START` | `false` | Sweep once at startup |
| `IGNORE_PATTERNS` | `*TdarrCache*,*.partial*,*.tmp,*-sample.*,*.sample.*,.*` | Glob patterns for names to skip |
| `PATH_MAPPINGS` | | `from:to,...` for paths coming in via hooks |
| `PUID` / `PGID` / `UMASK` | `99` / `100` / `002` | File ownership of rewritten subtitles |

## Statuses

`fixed` timestamps rewritten · `in_sync` checked, no change needed · `needs_fix` (dry run) would be fixed · `uncertain` couldn't decide confidently, left untouched · `embedded` only embedded subtitles, not checked (`EMBEDDED_SUBS=report`) · `no_subs` no subtitles (or only image-based ones) · `failed` error (see details) · `missing` file gone.

## Undo
Originals are saved under `/config/backups/<date>/<original path>.<time>`. Copy one back to restore it.

## Development
```bash
go test ./...
go run ./cmd/captionsync   # needs ffmpeg + whisper-server-{cpu,cuda} in WHISPER_BIN_DIR for processing
```
