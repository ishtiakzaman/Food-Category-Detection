#!/bin/bash
for dir in `ls train`;
do
	mkdir -p siftcache/train/$dir
done

for dir in `ls test`;
do
	mkdir -p siftcache/test/$dir
done

