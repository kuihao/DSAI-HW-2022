# DSAI-HW-2022
The homework-1 of the [NCKU](https://www.ncku.edu.tw/index.php?Lang=en) course which named Competitions in [**D**ata **S**ciences and **A**rtificial **I**ntelligence](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0110&sem=2&co_no=P75J000&class_code=).

## PROBLEM DESPRICTION
### TASK
In this HW, we will implement an algorithm to predict the **operating reserve (備轉容量)** of electrical power. Given a time series electricity data to predict the value of the operating reserve value of **each day during 2022/03/30 ~ 2022/04/13.**
### LIMITION
(Claimed by teaching assistant at 2022/03/22, Tue, 16:37 in NCKU Moodle)
* 訓練階段中，訓練資料集只能使用 2022 年 3 月 30 日 00:00 前的資料
* 測試階段中，輸入資料只要是當天或以前即可 （舉例：要預測 4/10 的備載容量，輸入資料只能是 4/10 或以前）
* Prohibited Matter (禁止事項): You can not use these data as training data or your answer.
    * ![Prohibited Matter MEME](https://pic.pimg.tw/merfolk/4a0276872c9d2.jpg)
    * [今日預估尖峰備轉容量率](https://www.taipower.com.tw/tc/page.aspx?mid=206&cid=405&cchk=e1726094-d08c-431e-abee-05665ab1c974)
    * [未來一週電力供需預測](https://www.taipower.com.tw/tc/page.aspx?mid=209)
    * [未來一週的天氣預報](https://opendata.cwb.gov.tw/dataset/forecast/F-A0010-001)
    * [未來三日空氣品質預報](https://airtw.epa.gov.tw/CHT/Forecast/Forecast_3days.aspx)
* The code `app.py` must be legally executed or output the `submission.csv` correctly.
* Repo. must have `requirement.txt`.
* Your method must be ML/DR method, otherwise the maximum point will be 60.
* Your code cannot be the same as other classmate’s.
### Grading (評量方式)
Root-Mean-Squared-Error of student's algorithm's predictions and answers in reality.
## INSTRUCTIONS FOR USE (使用說明)
* (TODO) 設計接口，允許 TA 放入原始的測試資料，並執行測試階段
## **HIGHLIGHT**
### OBSERVATION
* 備轉容量和備用容量是不同的
    * 本作業要預測的是備轉容量<br>
      **備轉容量 = 系統運轉淨尖峰能力 (當天最大發電容量)－系統瞬時尖峰負載 (當天「瞬間」最高用電量)**
      > 備轉容量 (Operating Reserve) 指當天實際可調度之發電容量裕度，亦即系統每天的供電餘裕。([ref.](https://www.taipower.com.tw/tc/page.aspx?mid=206&cid=405&cchk=e1726094-d08c-431e-abee-05665ab1c974))
      * [英文對照](https://www.taipower.com.tw/en/page.aspx?mid=4484&cid=2833&cchk=083f3aa1-77b0-43cf-9e4f-877a8a484c39): Operating Reserve = Net Peaking Capability - Peak Load (instantaneous value)
    * [備用容量 (Reserve Margin)](https://www.taipower.com.tw/tc/page.aspx?mid=212&cid=118&cchk=2b7682d9-46f8-4103-b636-02a5afeda67c)
* 
    ### PRE-PROCESSING
    * 由 FFT 可以發現，顯然 Net Peaking Capability 與 Peak Load 除了年呈現週期性變化之外，<br>
    每一周、每 3.5天、每 2.5 天 (推測是假日或節期)，也會呈現週期性變化
    <br>直接對照真實日期會發現大致上 (週日至週一) 的 Peak Load 會特別低、(週二至週六) 的 Peak Load 會特別高
    <br>雖然不知為何台電不直接顯示 Net Peaking Capability 趨勢，但從本實驗中可發現 Net Peaking Capability 大<br>致上是跟著 Peak Load 變化
    畢竟如果是用電的人少的日子，也不用開這麼多機組待命?
    <br>因此由 RNN 行訓練時，因將 7天、3.5天、2.5天 各視為一個合理的變化區段，或說預期 RNN 能找到每周固定的變化規律
    <br>如此一來， input feature 除了前一周的 Net Peaking Capability、Peak Load，也應該包含星期幾
    * **刪除離群值、歧異值、級特殊事件、與現況不符等的時間區段**
      * 年假之全國大停工期間，資料刪去: 
        * 2022/1/25 ~ 2022/2/7 
        * 2021/2/9 ~ 2021/2/17
        * 2020/1/21 ~ 2020/2/3
        * 2019/1/29 ~ 2019/2/11
      * [2021、2022 大停電](https://zh.wikipedia.org/wiki/%E5%8F%B0%E7%81%A3%E5%A4%A7%E5%81%9C%E9%9B%BB%E5%88%97%E8%A1%A8)資料刪去
        * 2021 年 5 月 13 日 (2021/5/11 ~ 2021/5/17)
        * 2021 年 5 月 17 日 (2021/5/11 ~ 2021/5/17)
        * 2022 年 3 月  3 日 (2022/3/1 ~ 2022/3/7)
      * [2021 年中華民主共和國 (Republic of China) 全國疫情第三級警戒](https://zh.wikipedia.org/wiki/2021%E5%B9%B4%E4%B8%AD%E8%8F%AF%E6%B0%91%E5%9C%8B%E5%85%A8%E5%9C%8B%E7%96%AB%E6%83%85%E7%AC%AC%E4%B8%89%E7%B4%9A%E8%AD%A6%E6%88%92)，現在疫情趨向穩定，與三級緊戒疫情居家辦公型態不同，資料刪去
        * 2021/5/15 ~ 2021/7/27 (三級降二級)
        * 2021/10/08 紙本振興五倍券第一批開放領取
        * [2021/10/27 我國疫苗第一劑注射率達 70%、第二劑達 30%](https://covid19.mohw.gov.tw/ch/sp-timeline0-205.html) 表示疫情逐漸趨緩
        * 刪去 (2021/5/11 ~ 2021/11/1)
### 訓練資料與驗證資料集的選擇
本次預測期間為含蓋清明節 (春假) 之前後兩周間，故選擇以 2020、2021 的清明節附近兩周為驗證資料。
* 2021/3/30 ~ 2021/4/13 (清明節為 4/4)
* 2020/3/31 ~ 2020/4/14 (清明節為 4/4)
其餘皆為訓練資料 (2019/01/01 ~ 2022/02/28、2022/03/29?)

### 透過快速傅立葉轉換觀察數據週期性
本實驗一共蒐集 2019 ~ 2022 大約 3 年多的資料，值觀可發覺數據有逐年攀升的趨勢，且明顯存在兩種主要頻率
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/TimeDomain_1.png?raw=true)
我又透過 Tensorflow 提供的實數快速傅立葉轉換進行分析，並得到確信變化的頻率：
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/FreqDomain_NPC.png?raw=true)
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/FreqDomain_PL.png?raw=true)
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/FreqDomain_OR.png?raw=true)
由上圖可知，Net Peaking Capability 與 Peak Load 確實非常相似，並且在頻域中可見 1/周、1/3.5天、1/2.3天，存在較為明顯的週期性變化。對照實際日期後會發現，每當「周日、周一」時，Net Peaking Capability 與 Peak Load 會特別低；反之其他日子，Net Peaking Capability 與 Peak Load 都有攀升現象。因此，可見「星期幾」是相當重要的因素之一。

### **2+2 GRU and 1 DNN 混合式架構**
本次實驗共使用 3 個模型進行預測，
* 模型一
![Main-model](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/model_blockimg.png?raw=true)
  * 使用兩個 GRU (一個專門學習 Net Peaking Capability 的變化趨勢、一個專門學習 Peak Load 的變化趨勢)，兩 GRU 合併時是由 Net Peaking Capability GRU 的輸出減去 Peak Load GRU 的輸出，以此模擬台電所聲稱的公式。
  * 同時，我還加入第三個 DNN 輸入，此為模型擴充微調的部分，可以於此處加入當天的氣溫、雨量、溫溼度...等資訊，二度強化模型。但因為時間因素，本次只有加入星期的特徵。
  * 此模型會根據前 7 天的 Net Peaking Capability 與 Peak Load、星期、特殊日子的電力特徵來輸出未來 1 天的 Operating Reserve.
* 模型二與三:
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/model2_blockimg.png?raw=true)
  * 模型二是我再利用另外一個 GRU 模型，學前 7 天的 Net Peaking Capability 變化，輸出未來一天的  Net Peaking Capability 
  * 模型三是我再利用另外一個 GRU 模型，學前 7 天的 Peak Load 變化，輸出未來一天的 Peak Load
透過三個大模型的交互作用來得出最終的備轉容量 (Operating Reserve)。

## LISTING
(此處為簡要說明各檔案的功能)

## ENVIRONMENT AND EXPERIMENT DESIGN
* Python: 3.6.4
* OS: Windows 10
* GPU: Nvidia Geforce GTX 1070
* Dataset
    * [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995) *(Download date: 2022/03/27)*
        * 2021/01~2022/02: 透過 `CSV 按鈕` 取得 
        * 2020/01~2021/04: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕` 取得
        * 2019/01~2021/01: 從[去年學長姐的 repositorie (vf19961226)](https://github.com/vf19961226/Electricity-Forecasting/blob/main/data/%E5%8F%B0%E7%81%A3%E9%9B%BB%E5%8A%9B%E5%85%AC%E5%8F%B8_%E9%81%8E%E5%8E%BB%E9%9B%BB%E5%8A%9B%E4%BE%9B%E9%9C%80%E8%B3%87%E8%A8%8A.csv) 下載取得
    * [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850) *(Download date: 2022/03/27)*
        * 2022: 透過 `CSV 按鈕` 取得
        * 2021: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕` 取得
    * [台灣電力公司_近三年每日尖峰備轉容量率](https://data.gov.tw/dataset/24945) *(Download date: 2022/03/27)*
      * 紀錄 2019/1/1 ~ 2021/12/31 每日尖峰的「備轉容量(萬瓩)」與「備轉容量率(%)」
    * [當天天氣](https://opendata.cwb.gov.tw/dataset/observation/O-A0003-001)(來不及使用) 
    * [當天空氣品質](https://data.gov.tw/dataset/40448)(來不及使用)
    * [台灣電力公司_未來兩年機組大修停機排程](https://data.gov.tw/dataset/35393) *(Download date: 2022/03/27)*
      * 2022~2023: 透過 `CSV 按鈕` 取得 
      * 2021~2022: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕` 取得
    * [台電歷年尖峰負載及備用容量率](https://data.gov.tw/dataset/8307) *(Download date: 2022/03/27)*
    * [台灣電力公司_各年度再生能源別裝置容量](https://data.gov.tw/dataset/29933) *(Download date: 2022/03/27)*
      * [裝置容量](https://smctw.tw/4223/) 是指該設備出廠時，所設計滿載（百分之百全力發電）時的最大值，不同的設備常用的單位會不同。
* Random seed: 
## TRAINING AND TUNING (訓練階段)
平均每次模型都會訓練上千個 epoch，起初會將學習率設為 10，加速梯度下降速度，較後期會將學習率逐漸調小。**(由於模型眾多且龐大、訓練時間非常冗長，訓練過程需要手動中斷等等，本實驗已先將模型訓練完畢並儲存於資料夾，app.py 會直接讀取模型存檔，為保留公平原則，一切實驗方法都保存於 app.ipynb 以供查驗，絕無犯規作弊。)**
![](https://github.com/kuihao/DSAI-HW-2022/blob/main/log/img/training_stage_2_lr1e2.png?raw=true)

## 最佳模型結果:
本實驗的驗證集採用與最終評分相似的 2020、2021 年的清明節前後兩周作為評估：
```python 
m = tf.keras.metrics.RootMeanSquaredError()
m.update_state(Pred_valid, [GroundTruth_valid])
m.result().numpy()
```
**得出的 Operating Reserve RMSE 結果為: 423.14825**
## EVALUTION (測試階段)
(等待評分結果)