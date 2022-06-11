#!/bin/bash
#

#read -p "Input the files path: " paths
paths="$@"
path2="./train525/"
total=0
echo "==$paths=="
for i in  `ls $paths`
do
#       echo "$i:"
        printf "\033[33m${i}:\033[0m \t" 
        num=`ls ${paths}/$i | wc -l`
	echo "$num"
	let total=($num+$total)
done
echo "total:$total"
