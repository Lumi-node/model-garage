#!/bin/bash
# Record a terminal demo of Model Garage CLI
# Outputs: assets/demo.cast (asciinema recording)
#
# To convert to GIF:
#   pip install agg  (or use https://github.com/asciinema/agg)
#   agg assets/demo.cast assets/demo.gif --theme monokai
#
# To play back:
#   asciinema play assets/demo.cast

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CAST_FILE="$REPO_DIR/assets/demo.cast"

cd "$REPO_DIR"

echo "Recording demo to $CAST_FILE ..."
echo "This will run actual garage commands. Press Ctrl+C to abort."
echo ""

# Use asciinema's scripted recording
asciinema rec "$CAST_FILE" --overwrite -c "python scripts/demo_session.py"

echo ""
echo "Done! Recording saved to $CAST_FILE"
echo ""
echo "Convert to GIF with:"
echo "  agg $CAST_FILE assets/demo.gif --theme monokai --cols 80 --rows 30"
