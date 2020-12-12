#!/bin/bash

N=0
for f in $(find ./bitmoji); do
	convert $f -background white -flatten ./wout/$N.png
	N=$((N+1))
done
