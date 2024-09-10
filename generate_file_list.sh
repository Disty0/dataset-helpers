#!/bin/bash

mkdir ../image_list
mkdir ../text_list

echo "VNCG WD"
for i in vncg/vncg/*/*.jpg; do echo "$i" >> ../image_list/vncg_wd_image_list.txt; done
for i in vncg/vncg/*/*.txt; do echo "$i" >> ../text_list/vncg_wd_text_list.txt; done

echo "VNCG NL"
for i in vncg/vncg-nl/*/*.jpg; do echo "$i" >> ../image_list/vncg_nl_image_list.txt; done
for i in vncg/vncg-nl/*/*.txt; do echo "$i" >> ../text_list/vncg_nl_text_list.txt; done

echo "Booru Tag"
for i in booru/booru/*/*.webp; do echo "$i" >> ../image_list/booru_tag_image_list.txt; done
for i in booru/booru/*/*.txt; do echo "$i" >> ../text_list/booru_tag_text_list.txt; done

echo "Booru NL"
for i in booru/booru-nl/*/*.webp; do echo "$i" >> ../image_list/booru_nl_image_list.txt; done
for i in booru/booru-nl/*/*.txt; do echo "$i" >> ../text_list/booru_nl_text_list.txt; done
