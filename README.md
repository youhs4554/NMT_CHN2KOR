# NMT_CHN2KOR
- NMT development for CHN2KOR
- For big-data analysis assignments
- Named entity(or proper nouns) and normal words are separated
- Embeddings of Word(proper nouns) + Character(others) are merged

# Requirements
- Python 2.7.12, virtualenvs
- GPU : CUDA 8.0, CUDNN 6.0.20
- If GPU is not available, install tensorflow cpu version, replace 'tensorflow-gpu' with 'tensorflow' in requirements.txt

    ## Package install
    > pip install -r requirements.txt
    
    ## Pretrained Model
    > https://drive.google.com/open?id=1mBsNqaaRLZtLzHjhiocJV9qe57_wyIDU
    
    ## Dataset
    - raw parallel corpora (CHN-KOR)
    - train/valid/test
    > https://drive.google.com/open?id=1taSYAJE2hbRhTKEE7bACOEtKL54jg5aZ
    
    ## Results File
    - hypo/gold
    - .json file, better view in web browser
    > https://drive.google.com/open?id=16FBfjb-nJOlqOpg49-zHltGKuRth4155
    
# Usage
>After download above required files, extract *.tar.gz
    
    # extract *.tar.gz
    cat *.tar.gz | tar -xvzf - -i

>Supports, 3 mode(train/eval/demo)

    # train model
    python main.py --mode=train --gpu=0 --batch_size=64

    # eval model
    python main.py --mode=eval --gpu=0 --batch_size=1

    # demo
    python main.py --mode=demo --gpu=0 --batch_size=1

>Performance Measure
    
    python NMTEval_scores.py
>>Results  
- {'Bleu_4': 0.25607079605073346, 'Bleu_3': 0.32166887220324386, 'Bleu_2': 0.4097237861183638, 'Bleu_1': 0.5375119923065272, 'ROUGE_L': 0.5582416304146353, 'METEOR': 0.30310215444761324}

    
# Further works
- Hierarchical embedding (stroke->char->word)
- Hierarchical RNN structure
- Hanzi node.js package link
    > https://github.com/nieldlr/hanzi