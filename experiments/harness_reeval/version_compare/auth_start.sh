#!/usr/bin/env bash
# Start `colab sessions` (the auth trigger) as a detached process whose stdin is
# a kept-open FIFO, so we can: (1) read the OAuth URL it prints, hand it to the
# user, then (2) write the authorization code back into the SAME process later
# (PKCE requires same-process). Output goes to /tmp/authout.
set -e
# kill any stale colab auth processes so only ONE challenge is live
pkill -f "colab sessions" 2>/dev/null || true
rm -f /tmp/authout /tmp/authpipe
mkfifo /tmp/authpipe
export OAUTHLIB_RELAX_TOKEN_SCOPE=1   # tolerate scope-order differences from Google
# open the fifo read-write inside the child so it never EOFs while colab waits
setsid bash -c 'export OAUTHLIB_RELAX_TOKEN_SCOPE=1; exec 3<>/tmp/authpipe; colab sessions <&3 >/tmp/authout 2>&1' &
# wait for the URL to appear
for _ in $(seq 1 50); do
  if grep -q "accounts.google.com" /tmp/authout 2>/dev/null; then break; fi
  sleep 0.3
done
echo "=================== OAUTH URL ==================="
cat /tmp/authout
echo "================================================="
