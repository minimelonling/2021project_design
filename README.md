# 專題

## 2021.05.23

### 專題目標

### 程式架構
```
 2021project_design
 --- fit_map.py
 --- input
      --- map
      --- draw
 --- output
 --- backup
 --- convert_binary
     --- make_binary.py
     --- draw
     --- map
     --- training_data
         --- input
         --- output
```

### 程式說明

#### fit_map.py

* 位置: 主資料夾底下
* 執行指令
```
python fit_map.py --map {map} --draw {draw} --output {output}
```
* 輸入
map: `input/map`底下的地圖圖片(道路須為白色)
draw: `input/draw`底下的使用者繪畫線條圖片
output: 輸入output檔名(程式結束後會輸出到`output`資料夾底下)
* 輸出
與原本地圖合併的輸出，會輸出到`output`底下
只有路線的輸出，會輸出到`convert_binary/training_data/output底下`

```
繪畫圖片輸入注意事項:
1. 背景須為白色
2. 畫筆顏色不可跟白色太相近
3. 只能有簡單的一筆畫線條 (因為太複雜跑不出來，陽春的rule base QQ)
```
* 程式運作原理
 
#### make_binary.py

* 位置: convert_binary底下
* 執行指令
```
python make_binary.py --binary {binary} --pairwise {pairwise} --base {base}
```
binary: `0`代表要輸出紅綠色，可看見的結果，`1`代表要輸出成binary形式
pairwise: `0`代表每一個地圖都會跟每一個線條圖案匹配做輸出，`1`代表地圖集跟圖案集只會一對一批配
base: 本次輸出的檔案名稱要以base為基準做累加
* 輸入
程式會自動分別去讀取`convert_binary/map`跟`convert_binary/draw`底下的檔案
* 輸出
程式會自動輸出到`convert_binary/training_data/input`底下

* 程式運作原理


## 2021.05.19

the result is generated by using rule based code

hackmd:
https://hackmd.io/PacBNyzCTdGsrJ8JdI7-Pg?both

![](https://i.imgur.com/AZO8geV.png)
![](https://i.imgur.com/n3U960z.png)

![](https://i.imgur.com/ZBlOyJu.png)
![](https://i.imgur.com/nWd8XhF.png)

![](https://i.imgur.com/019Z0XG.png)
![](https://i.imgur.com/cJeKqE2.png)

![](https://i.imgur.com/T7v79PL.png)
![](https://i.imgur.com/GrjZaAQ.png)

![](https://i.imgur.com/GxkkVwE.png)
![](https://i.imgur.com/KKPCLbM.png)

![](https://i.imgur.com/QEvGw22.png)
![](https://i.imgur.com/X8NJxis.png)

![](https://i.imgur.com/QcrrAN1.png)
![](https://i.imgur.com/xmg7YHY.png)

![](https://i.imgur.com/R0ENnZB.png)
![](https://i.imgur.com/u9Jn9fT.png)

![](https://i.imgur.com/Ra3w4yw.png)
![](https://i.imgur.com/9IuaahQ.png)

![](https://i.imgur.com/OswY5xa.png)
![](https://i.imgur.com/dldX9X1.png)

![](https://i.imgur.com/IUtowuv.png)
![](https://i.imgur.com/pSwY5e5.png)

![](https://i.imgur.com/9shAlHa.png)
![](https://i.imgur.com/Xf4Cpim.png)

![](https://i.imgur.com/vE4pkL8.png)
![](https://i.imgur.com/DzxRzji.png)

![](https://i.imgur.com/u8K3CgI.png)
![](https://i.imgur.com/f1gEDF0.png)

![](https://i.imgur.com/NBIkcaY.png)
![](https://i.imgur.com/dnIxQHA.png)

![](https://i.imgur.com/KVJA5Xb.png)
![](https://i.imgur.com/08j3vkb.png)

![](https://i.imgur.com/fLiP5o7.png)
![](https://i.imgur.com/Ig0V2Jk.png)

![](https://i.imgur.com/c18zagw.png)
![](https://i.imgur.com/fOLy78y.png)



## 2021.04.16

???
![](https://i.imgur.com/recwRKe.png)


## 2021.03.28

讓使用者在白紙上任意塗鴉

![](https://i.imgur.com/BSdMdJR.png)

用model把圖片上的線條轉成區域

![](https://i.imgur.com/TFcBykq.png)

(變成區域再fit地圖比較可以覆蓋到地圖上的道路

把區域圖跟地圖疊合
(看看可不可以用影像處理的套件做到

![](https://i.imgur.com/66Gdrjs.png)

用model把畫上區域地圖轉成有線條的地圖

![](https://i.imgur.com/ZtlmEhU.png)


最後拿有線條(未完整)的地圖再跟原始地圖做疊合
得出最後成果
(一樣也是用套件

![](https://i.imgur.com/7fKyrtw.png)

其他例子><

fit前
![](https://i.imgur.com/LgAlYyR.png)

轉成區域
![](https://i.imgur.com/6K5QrnO.png)

fit後
![](https://i.imgur.com/p4H2Sp2.png)

xxxxxxxxxxxx


