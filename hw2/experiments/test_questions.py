"""任务2+4实验共用的测试问题集

每条问题标注:
- doc_id       : 答案所在文档(用于判断"是否命中正确来源")
- answer_span  : 文档中包含答案的一小段原文，用于自动判断检索到的chunk是否真的包含答案
                 (简单粗暴但足够本作业用: 判断answer_span是否出现在某个被检索到的chunk里)
- qtype        : "keyword" 表示题目偏精确关键词/数字匹配，预期BM25更强
                 "semantic" 表示题目用词和原文不同、需要语义理解，预期向量检索更强

这批题里特意设计了几个"答案位置刚好靠近段落中部/跨越明显停顿"的问题(标了boundary_risk=True)，
用来在chunking对比实验里观察fixed-size是否比sliding-window更容易漏检。
"""

QUESTIONS = [
    {
        "question": "远程办公每周最多能申请几天？",
        "doc_id": "doc1_remote_work",
        "answer_span": "每周最多可申请3天远程办公",
        "qtype": "keyword",
        "boundary_risk": False,
    },
    {
        "question": "在家办公需要保证网速吗？",  # 语义改写: "在家办公"≈"远程办公","网速"≈"网络带宽"
        "doc_id": "doc1_remote_work",
        "answer_span": "网络带宽不低于20Mbps",
        "qtype": "semantic",
        "boundary_risk": True,
    },
    {
        "question": "云帆智能手环P3无理由退货期限是多少天？",
        "doc_id": "doc2_return_policy",
        "answer_span": "自消费者签收商品之日起15日内",
        "qtype": "keyword",
        "boundary_risk": False,
    },
    {
        "question": "买的耳机坏了不是自己弄坏的，多久之内能免费修？",  # 语义改写: "坏了"≈"质量问题"
        "doc_id": "doc2_return_policy",
        "answer_span": "自签收之日起180日内均可申请免费换货或维修",
        "qtype": "semantic",
        "boundary_risk": True,
    },
    {
        "question": "新员工入职当天几点前要到公司？",
        "doc_id": "doc3_onboarding",
        "answer_span": "入职当天上午9点前抵达公司前台",
        "qtype": "keyword",
        "boundary_risk": False,
    },
    {
        "question": "带我熟悉工作的人多久换一次？",  # 语义改写: "带我熟悉工作的人"≈"导师"
        "doc_id": "doc3_onboarding",
        "answer_span": "导师制持续时间为入职后3个月",
        "qtype": "semantic",
        "boundary_risk": True,
    },
    {
        "question": "公司系统密码要求多少位？",
        "doc_id": "doc4_it_security",
        "answer_span": "密码长度不得少于10位",
        "qtype": "keyword",
        "boundary_risk": False,
    },
    {
        "question": "在星巴克用免费WiFi能处理涉密文件吗？",  # 语义改写: "星巴克免费WiFi"≈"公共WiFi"
        "doc_id": "doc4_it_security",
        "answer_span": "禁止处理任何标注为\"内部\"或\"机密\"级别的文件",
        "qtype": "semantic",
        "boundary_risk": True,
    },
    {
        "question": "VPN空闲多久会自动断开？",
        "doc_id": "doc4_it_security",
        "answer_span": "VPN连接空闲超过30分钟将自动断开",
        "qtype": "keyword",
        "boundary_risk": False,
    },
    {
        "question": "退货的运费谁出？",  # 语义/概括性问题，答案分两种情况，散落在段落中间
        "doc_id": "doc2_return_policy",
        "answer_span": "无理由退货产生的退货运费由消费者自行承担",
        "qtype": "semantic",
        "boundary_risk": True,
    },
]
