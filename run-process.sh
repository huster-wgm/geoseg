echo "Running Scripts for image proprocessing ..."
python ./utils/preprocess.py -data Vaihingen -mode slice -stride 112 -is_multi Yes &&
python ./utils/preprocess.py -data RS-2018-compress -mode both &&
python ./utils/preprocess.py -data LasVegas -mode both &&
echo "end"
