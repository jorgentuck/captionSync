#!/bin/sh
# Runs captionsync as PUID:PGID (Unraid default nobody:users = 99:100) so rewritten
# subtitle files keep the ownership your other containers expect.
set -e
PUID="${PUID:-99}"
PGID="${PGID:-100}"
umask "${UMASK:-002}"
mkdir -p "${CONFIG_DIR:-/config}"

if [ "$(id -u)" = "0" ] && [ "$PUID" != "0" ]; then
    if [ "$(stat -c %u "${CONFIG_DIR:-/config}")" != "$PUID" ]; then
        chown -R "$PUID:$PGID" "${CONFIG_DIR:-/config}" || true
    fi
    exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups /usr/local/bin/captionsync "$@"
fi
exec /usr/local/bin/captionsync "$@"
