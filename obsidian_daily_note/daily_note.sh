#!/bin/bash

# Obsidian Daily Note Creator
# Creates a daily note from template and cleans up unused yesterday's note

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="$SCRIPT_DIR/template.md"
OBSIDIAN_VAULT_PATH=""

TODAY=$(date +"%d-%m-%Y")
YESTERDAY=$(date -v-1d +"%d-%m-%Y")

TODAY_FILE="$OBSIDIAN_VAULT_PATH/$TODAY.md"
YESTERDAY_FILE="$OBSIDIAN_VAULT_PATH/$YESTERDAY.md"

if [ -z "$OBSIDIAN_VAULT_PATH" ]; then
    echo "Error: Obsidian Vault Path is not set."
    exit 1
fi

if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "Error: Template file not found at $TEMPLATE_PATH"
    exit 1
fi

if [ ! -f "$TODAY_FILE" ]; then
    cp "$TEMPLATE_PATH" "$TODAY_FILE"
    echo "Created today's note: $TODAY_FILE"
else
    echo "Today's note already exists: $TODAY_FILE"
fi


if [ -f "$YESTERDAY_FILE" ]; then
    if diff -q "$YESTERDAY_FILE" "$TEMPLATE_PATH" > /dev/null 2>&1; then
        rm "$YESTERDAY_FILE"
        echo "Deleted unused yesterday's note: $YESTERDAY_FILE"
    else
        echo "Yesterday's note has content, keeping: $YESTERDAY_FILE"
    fi
else
    echo "No yesterday's note found"
fi
