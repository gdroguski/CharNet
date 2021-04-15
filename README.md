# Convolutional Character Networks

This project hosts the testing code for CharNet, described in paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

Also I added Fast Non Maximum Suppression algorithm written in C++ and exported to dll to replace current vanilla postprocess (speedup ~30x) and made some code refactoring and added possibility to apply specified preprocess from config. The code for my Fast NMS is in other repo: [repo_link](https://github.com/gdroguski/FastNMS).

## How to

```
pip install -r requirements.txt
python setup.py build develop
pip install [torch, torchivsion]
```

1. For `some config`, please run the following command line in `config` dir. Please replace `images_dir` with the directory testing images. The results will be in `results_dir`.

    ```
    python predict.py configs/<some_config>.yaml <images_dir> <results_dir>
    ```
2. If there is WinError for importing a module *.dll it means that you don't have the C++ runtime available, to do so install vcredist or set `USE_CPP` to `False` in conifg 
