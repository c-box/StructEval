benchmark=$1
split=$2

python topic_extract.py --benchmark $benchmark --split $split
python bloom_generation.py --benchmark $benchmark --split $split
