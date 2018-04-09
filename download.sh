#!/usr/bin/env bash
FILEID='151Yz8bzjHuhftTbabs-0fKDfcw9fipGu'
FILENAME='data.tar.gz'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
tar -xvzf $FILENAME
mv chunk-100 data
rm $FILENAME