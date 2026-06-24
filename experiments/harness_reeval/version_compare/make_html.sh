#!/usr/bin/env bash
# Read the live OAuth URL from /tmp/theurl and write a clickable HTML file to the
# Windows Desktop, so the (very long) URL can't be truncated by copy-paste.
set -e
URL=$(cat /tmp/theurl)
OUT="/mnt/c/Users/z004knva/Desktop/colab_login.html"
{
  echo '<!doctype html><meta charset=utf-8><title>Colab login</title>'
  echo '<body style="font-family:sans-serif;padding:2em;font-size:18px;max-width:60em">'
  echo '<h2>Colab CLI login</h2>'
  echo '<p>1) Click the blue button. 2) Approve with your Colab-credit Google account (leave ALL boxes checked). 3) Copy the <b>4/0A...</b> code Google shows and paste it back to Claude.</p>'
  printf '<p><a href="%s" style="display:inline-block;padding:14px 22px;background:#1a73e8;color:#fff;text-decoration:none;border-radius:6px">Open Google login &rarr;</a></p>\n' "$URL"
  echo '<hr><p style="color:#666">If the button does nothing, copy this entire box into your browser address bar:</p>'
  printf '<textarea style="width:100%%;height:140px" readonly>%s</textarea>\n' "$URL"
  echo '</body>'
} > "$OUT"
echo "wrote $(wc -c < "$OUT") bytes to Desktop\\colab_login.html ; url length=${#URL}"
