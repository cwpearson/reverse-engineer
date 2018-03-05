#! /bin/bash

SCRIPT_PATH=script.r2
OUT_PATH=out.txt

LIB_PATH=$1
LIB_NAME=`basename "$LIB_PATH"`
shift

>&2 echo "Looking for r2 project:" $LIB_NAME


# create a new project if one isn't found
if r2 -p | grep -Fxq $LIB_NAME; then
    >&2 echo "Found project"
else
    >&2 echo "Creating new project with 1-time analysis..."
    >&2 r2 -c "aaa; Ps $LIB_NAME" -q "$LIB_PATH"
    >&2 echo "done"
fi

echo "" > "$SCRIPT_PATH"
for var in "$@"; do
    echo -n "s " >> "$SCRIPT_PATH"
    echo $var";" >> "$SCRIPT_PATH"
    echo "x 16 >> " "$OUT_PATH" >> "$SCRIPT_PATH"
done

r2 -p "$LIB_NAME" -i "$SCRIPT_PATH" -q;

