from typing import List, Union

class_labels = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "가", "거", "고", "구",
    "나", "너", "노", "누",
    "다", "더", "도", "두",
    "라", "러", "로", "루",
    "마", "머", "모", "무",
    "바", "배", "버", "보", "부",
    "사", "서", "소", "수",
    "아", "어", "오", "우",
    "자", "저", "조", "주",
    "하", "허", "호",
    "울",  # 서울
    "경", "기",
    "인", "천",
    "대", "전",
    "세", "종",
    "충", "남", "북",
    "강", "원",
    # 경남, 경북
    "산",  # 부산
    # 울산
    # 대구
    # 전남, 전북
    "광",  # 광주
    "제",  # 제주
]

def map_class_index_to_target(classes: List[str]) -> Union[List[str], None]:
    if len(classes) == len(class_labels):
        for i in range(len(classes)):
            classes[i] = class_labels[i]

        return classes
    else:
        return None
