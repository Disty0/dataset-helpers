#!/bin/bash

mkdir ../jpg_list
mkdir ../txt_list

echo "Anime Wallpapers"

for i in anime_wallpapers/*/*.jpg; do echo "$i" >> ../jpg_list/anime_wallpapers_jpg_list.txt; done
for i in anime_wallpapers/*/*.txt; do echo "$i" >> ../txt_list/anime_wallpapers_txt_list.txt; done


echo "Oldest"

for i in oldest/*/*.jpg; do echo "$i" >> ../jpg_list/oldest_jpg_list.txt; done
for i in oldest/*/*.txt; do echo "$i" >> ../txt_list/oldest_txt_list.txt; done

echo "Early"

for i in early/*/*.jpg; do echo "$i" >> ../jpg_list/early_jpg_list.txt; done
for i in early/*/*.txt; do echo "$i" >> ../txt_list/early_txt_list.txt; done

echo "Mid"

for i in mid/*/*.jpg; do echo "$i" >> ../jpg_list/mid_jpg_list.txt; done
for i in mid/*/*.txt; do echo "$i" >> ../txt_list/mid_txt_list.txt; done

echo "Recent"

for i in recent/*/*.jpg; do echo "$i" >> ../jpg_list/recent_jpg_list.txt; done
for i in recent/*/*.txt; do echo "$i" >> ../txt_list/recent_txt_list.txt; done

echo "Newest"

for i in newest/*/*.jpg; do echo "$i" >> ../jpg_list/newest_jpg_list.txt; done
for i in newest/*/*.txt; do echo "$i" >> ../txt_list/newest_txt_list.txt; done

echo "Pixiv"

for i in pixiv/*/*.jpg; do echo "$i" >> ../jpg_list/pixiv_jpg_list.txt; done
for i in pixiv/*/*.txt; do echo "$i" >> ../txt_list/pixiv_txt_list.txt; done

echo "VNCG"

for i in visual_novel_cg/*/*.jpg; do echo "$i" >> ../jpg_list/vncg_jpg_list.txt; done
for i in visual_novel_cg/*/*.txt; do echo "$i" >> ../txt_list/vncg_txt_list.txt; done

