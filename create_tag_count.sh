#!/bin/bash

echo "Tags"
sort out/tags_list.txt | uniq -c | sort -r >> out/tag_count.txt

echo "Words"
sort out/words_list.txt | uniq -c | sort -r >> out/word_count.txt
