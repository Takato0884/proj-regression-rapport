import sys
import logging
# Loggingの設定
# https://docs.python.org/ja/3/howto/logging.html#logging-basic-tutorial
def get_logger(level=logging.INFO):
    logging.basicConfig(filename="C:\\Users\\hayas\\proj-regression-general\\git\\output\\log\\20240603\\test.log", level=logging.INFO)
    log = logging.getLogger(__name__) # ロガーのインスタンス作成
    if log.handlers:
        return log
    log.setLevel(level) # 適切な出力先に振り分けられるべき最も低い深刻度を指定
    ch = logging.StreamHandler(sys.stdout) # ハンドラのインスタンス作成
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter) # ハンドラが使用するFormatterオブジェクトを選択
    log.addHandler(ch)
    return log
