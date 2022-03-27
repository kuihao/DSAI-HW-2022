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
    * [備用容量 (Reserve Margin)](https://www.taipower.com.tw/tc/page.aspx?mid=212&cid=118&cchk=2b7682d9-46f8-4103-b636-02a5afeda67c)
* 
    ### PRE-PROCESSING

## LISTING
## ENVIRONMENT AND EXPERIMENT DESIGN
* Python: 3.6.4
* OS: Windows 10
* GPU: Nvidia Geforce GTX 1070
* Dataset
    * [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995) *(Download date: 2022/03/27)*
        * 2021/01~2022/02: 透過 `CSV 按鈕`取得 
        * 2020/01~2021/04: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕`取得
        * 2019/01~2020/03: 從[去年學長姐的 repositorie(linzh0205)](https://github.com/linzh0205/-Electricity-Forecasting/blob/main/train.csv) 下載取得
    * [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850) *(Download date: 2022/03/27)*
        * 2022: 透過 `CSV 按鈕`取得
        * 2021: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕`取得
    * [台灣電力公司_近三年每日尖峰備轉容量率](https://data.gov.tw/dataset/24945) *(Download date: 2022/03/27)*
      * 紀錄 2019/1/1 ~ 2021/12/31 每日尖峰的「備轉容量(萬瓩)」與「備轉容量率(%)」
    * [當天天氣](https://opendata.cwb.gov.tw/dataset/observation/O-A0003-001) 
    * [當天空氣品質](https://data.gov.tw/dataset/40448)
    * [台灣電力公司_未來兩年機組大修停機排程](https://data.gov.tw/dataset/35393) *(Download date: 2022/03/27)*
      * 2022~2023: 透過 `CSV 按鈕`取得 
      * 2021~2022: 透過檢視資料 > 多元格式參考資料 > `CSV 按鈕`取得
    * [台電歷年尖峰負載及備用容量率](https://data.gov.tw/dataset/8307) *(Download date: 2022/03/27)*
    * [台灣電力公司_各年度再生能源別裝置容量](https://data.gov.tw/dataset/29933) *(Download date: 2022/03/27)*
      * [裝置容量](https://smctw.tw/4223/) 是指該設備出廠時，所設計滿載（百分之百全力發電）時的最大值，不同的設備常用的單位會不同。
* Random seed: 
## TRAINING AND TUNING (訓練階段)
## EVALUTION (測試階段)