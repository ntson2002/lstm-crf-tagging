# Thông tin dữ liệu

## JSON format 
Thư mục json chứa dữ liệu. Mỗi tập tin chứa danh sách các câu. Thông tin mỗi token bao gồm: Head word, POS, CHUNK, Gold label của layer 1, Gold label của layer 2 

```
[

    [
        [
            "Chị",
            "Ns",
            "B-NP",
            "O",
            "O"
        ],
        [
            "Minh",
            "NNP",
            "B-NP",
            "B-PER",
            "O"
        ],
        [
            "ôm",
            "V",
            "B-VP",
            "O",
            "O"
        ]
    ]
]
```

Thư mục:

* dev (27 file)* testa (26 file)* testb (45 file)* train (212 file)


## CONLL format 
Data tự chia lại:

* train: được lấy từ tập dev  (212 file)
* dev: được lấy từ tập dev (27 file)
* testa: được lấy từ tập dev (26 file)

* testb; official test (45 file)

Thư mục:

* conll-1layer: train, dev, testa, testb cho layer đầu tiên
* conll-2layer: train, dev, testa, testb cho cả 2 layer 
* conll-1layer-crf: Gộp train và dev của các bộ dữ liệu trên thành dữ liệu huấn luyện cho CRF. Dữ liệu testa, testb không đổi
	* testa.conll	* testb.conll	* train.conll
* conll-2layer-crf: Dữ liệu bao gồm 2 layer 
