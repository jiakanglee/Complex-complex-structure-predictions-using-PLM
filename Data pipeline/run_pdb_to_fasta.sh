#! /bin/bash

cd /lijiakang/local_project/

export PATH=/opt/anaconda3/bin:$PATH
source $(which activate) LTEnjoy

/opt/anaconda3/envs/bin/
python parse_pdb.py
