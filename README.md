瞬間表情力テスト
==

## 概要
自分の顔の周りが青い矩形で囲まれます、黄色（Happy）かオレンジ（Angry）の矩形も表示されます。  
その矩形の枠の中に色に合わせた表情で顔を映してください。  
この映すスピードと表情力（何％の割合で表情を再現できているか）をスコア化します。  

## ルール
* 制限時間は1分間
* スコアが高ければ高いほど良い（瞬発表情力がある）
* 1人用
* カメラに向かって正面に顔を映さないと認識されないことがあるので正面に立ちましょう

## 使い方
### 使用ライブラリ
* NumPy
* cv2(OpenCV)
* TensorFlow(Kerasも含める)

### 【重要】EmoPyのコードの変更  
予測値が返ってくるように少し改良させる。ファイルパスは環境によって異なるが、anacondaを使っている場合は下のパス通りのところにあるはず。
>"
/home/username/anaconda3/envs/deep36/lib/python3.6/site-packages/EmoPy/src/fermodel.py  
60行目  
        # self._print_prediction(prediction[0]) # past  
        memo = self._print_prediction(prediction[0])  # add  
        return memo  # add  
60行目をコメントアウトし、後ろの2行を追加。  
113行目に  
        return [str(dominant_emotion), normalized_prediction[self.emotion_map[emotion]] * 100]  
を追加。  
実行結果が返ってくるように改良した。"  

main.pyから引用  
