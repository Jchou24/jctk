# stanford_corenlp
The core of stanford_corenlp is a toolkit clone from [dasmith/stanford-corenlp-python](https://github.com/dasmith/stanford-corenlp-python) on github
This toolkit makes some method for ease to use standford parser

# Usage
Since loading the corpus model of stanford parser well require secs. Launch the server of standford parser will decrease the parsing time.

    cd jctk/stanford_corenlp
    python corenlp.py

If you use the parser without launching the server, the parser will automatically load the corpus model. But this will increase the run time of your program.
