"""Communication policy: the tables behind docs/syrus/LEDGER_REVIEW.md. Draft wording; the Vietnamese text needs owner review.

Everything the server decides about an answer lives here: which findings a purpose needs, how many claim units a depth
allows, which qualifications a claim triggers, and which phrasings are refused. The model chooses purposes and writes
cited units; the server owns the rest.
"""
import re

DEPTHS = ["quick", "investigation", "technical"]
DEPTH = {"quick": {"units": 3, "formulas": False}, "investigation": {"units": 6, "formulas": False}, "technical": {"units": 10, "formulas": True}}

# purpose -> default depth and the findings an answer with that purpose must cover (finding, minimum depth).
# A required finding that does not fire for the case is not required.
PURPOSES = {
    "what_happened": {"depth": "quick", "require": [("F-MARGIN", "quick"), ("F-EWMA-MEMORY", "quick"), ("F-EVIDENCE-TIMING", "investigation")]},
    "possible_causes": {"depth": "investigation", "require": [("F-CANDIDATES", "quick"), ("F-IDENT", "quick"), ("F-CF-NONUNIQUE", "investigation")]},
    "evidence_and_sources": {"depth": "quick", "require": []},
    "method": {"depth": "technical", "require": []},
    "outside_evidence": {"depth": "quick", "require": []},
}
# A follow-up question is offered only when a purpose that compares candidates was declared.
QUESTION_PURPOSES = {"possible_causes"}

# refusal category -> record IDs of which at least one must be cited, and qualifications forced onto the first unit.
REFUSALS = {
    "diagnosis_request": {"cite": [], "quals": ["score_not_diagnosis"]},
    "causal_claim": {"cite": [], "quals": ["share_not_cause"]},
    "prediction": {"cite": [], "quals": ["precursor"]},
    "energy_or_cost": {"cite": ["bridge.status", "F-BRIDGE"], "quals": ["real_sim"]},
    "simulated_transfer": {"cite": ["bridge.status", "F-BRIDGE"], "quals": ["real_sim"]},
    "missing_document": {"cite": ["unavailable:documents"], "quals": []},
    "missing_data": {"cite": ["unavailable:raw_series", "unavailable:definitions", "unavailable:demand_measurement"], "quals": []},
    "operator_claim": {"cite": [], "quals": []},
    "instruction_override": {"cite": [], "quals": []},
    "out_of_scope": {"cite": [], "quals": []},
}

# Text that names an absent source must cite the matching unavailable record.
TOPICS = [
    ("documents", r"\b(?:manuals?|sops?|datasheets?|oem|maintenance (?:logs?|records?|tickets?)|sổ tay|nhật ký bảo trì|sổ bảo trì)\b", "unavailable:documents"),
    ("raw_series", r"raw (?:traces?|telemetry|waveforms?|series|signals?)|waveforms?|dữ liệu thô|đồ thị thô", "unavailable:raw_series"),
    ("definitions", r"defined as|definition of|stands for|định nghĩa|là viết tắt", "unavailable:definitions"),
]

# Phrasings refused in unit text. Entries marked exempt are allowed when a negation precedes them ("is not a diagnosis").
FORBIDDEN = [
    ("cause", r"\bcaused\b|\bcausing\b|root[- ]cause|\bcause of\b|\bthe cause\b|responsible for|\bculprit\b|nguyên nhân (?:gốc|chính|là)|gây ra", True),
    ("diagnosis", r"\bdiagnos(?:is|es|ed|e)\b|chẩn đoán", True),
    ("probability", r"\bprobab\w*|\blikel(?:y|ihood)\b|\bchance\b|\bconfidence\b|xác suất|khả năng cao|độ tin cậy", True),
    ("certainty", r"\bconfirmed?\b|\bdefinitely\b|\bcertainly\b|\bproves?\b|\bproved\b|chắc chắn|xác nhận", True),
    ("leak_assertion", r"\b(?:is|was|were|are|been) (?:an? |the )?(?:air )?leak\b|\bleak (?:was|is) (?:present|detected)|\bleak detected\b|\bdetected an? (?:air )?leak\b|phát hiện (?:một )?(?:chỗ )?rò rỉ", False),
    ("prediction", r"\bpredict\w*|\bforecast\w*|early warning of|dự đoán|dự báo", True),
    ("status", r"\bfalse alarms?\b|\bhealthy\b|normal operation\b|\bmissed fault\b|báo động giả|khỏe mạnh", True),
    ("energy", r"\bk?wh\b|\bkw\b|kg/s|\bkg\b|\bm3\b|\bjoules?\b|\bsavings?\b|\bsaved\b|\bcosts?\b|\busd\b|\beuros?\b|[$€]|tiết kiệm|chi phí|điện năng", True),
    ("procedure", r"\byou (?:should|must|need to)\b|\brecommend\w*|\b(?:shut|turn) (?:down|off)\b|\bstep[- ]by[- ]step\b|\bprocedures?\b|bạn nên|khuyến nghị|quy trình", True),
    ("ranking", r"\bmost (?:likely|probable|plausible)\b|\b(?:more|less) likely\b|\b(?:top|leading|primary) (?:candidate|hypothesis|explanation)\b|khả năng nhất", False),
    ("internal_id", r"\b(?:alert|contrib|calrank|cf|hyp|temporal|measured|bench|label|case|detector|unavailable|bridge|group)[.:][A-Za-z0-9_.]+|\bF-[A-Z-]+\b", False),
]
HYPOTHESIS_VERBS = r"\b(?:causes?|causing|caused|explains?|produces?|results? in|leads? to)\b|gây ra|dẫn đến"
NEGATION = re.compile(r"\b(?:not|no|never|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|without|nor|neither|rather than|instead of)\b|không|chưa|chẳng|đừng", re.IGNORECASE)
# Background units say nothing about this case, its signals or numbers.
CASE_REFERENCE = r"\b(?:this|the) (?:alert|case|incident|machine|compressor|train|detector|score|window)\b|cảnh báo này|ca này|máy nén|tàu"
FORMULA = re.compile(r"[=×÷]")
VIETNAMESE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")

# Qualifications. `ids` are matched against the record and finding IDs a unit cites directly (exact, or prefix when it ends
# with a dot); `text` is matched against the unit text in English and Vietnamese. Critical ones are shown whenever triggered
# and shortened after their first appearance in a conversation; context ones are shown once per conversation.
QUALS = [
    {"id": "score_not_diagnosis", "severity": "critical", "ids": ["alert.smoothed", "alert.threshold", "alert.margin", "F-MARGIN"], "text": r"threshold|ngưỡng",
     "en": "A score above the threshold means the data departed from the pattern learned in training; it is not a fault diagnosis.",
     "vi": "Điểm vượt ngưỡng cho thấy dữ liệu lệch khỏi mẫu đã học khi huấn luyện; đó không phải chẩn đoán lỗi.",
     "short_en": "Above the threshold is not a fault diagnosis.", "short_vi": "Vượt ngưỡng không phải chẩn đoán lỗi."},
    {"id": "share_not_cause", "severity": "critical", "ids": ["contrib.", "calrank."], "text": r"\bshare\b|percentile|\brank|tỷ trọng|phân vị|xếp hạng",
     "en": "A share or rank describes the detector's evidence, not a probability or a physical cause.",
     "vi": "Tỷ trọng hay thứ hạng mô tả bằng chứng của bộ phát hiện, không phải xác suất hay nguyên nhân vật lý.",
     "short_en": "Detector evidence, not a probability or cause.", "short_vi": "Bằng chứng của bộ phát hiện, không phải xác suất hay nguyên nhân."},
    {"id": "repair_tests_detector", "severity": "critical", "ids": ["cf.", "detector.repair", "F-CF-NONUNIQUE", "F-CF-PERSIST"], "text": r"repair|replac|reconstruct|sửa|thay thế|tái tạo",
     "en": "These repairs test the detector, not the machine; they do not show which group caused the alert or what a sensor fix would do.",
     "vi": "Các phép sửa này kiểm tra bộ phát hiện chứ không phải máy; chúng không cho biết nhóm tín hiệu nào gây ra cảnh báo hay việc sửa cảm biến sẽ ra sao.",
     "short_en": "Tests the detector, not the machine.", "short_vi": "Kiểm tra bộ phát hiện, không phải máy."},
    {"id": "timing_not_onset", "severity": "critical", "ids": ["temporal.", "F-EVIDENCE-TIMING"], "text": r"evidence (?:rose|emerged)|bằng chứng (?:tăng|xuất hiện)",
     "en": "These rank the detector's smoothed evidence; they do not show when a fault began.",
     "vi": "Đây là thứ hạng bằng chứng đã làm mượt của bộ phát hiện; chúng không cho biết lỗi bắt đầu khi nào.",
     "short_en": "Smoothed evidence, not fault onset.", "short_vi": "Bằng chứng đã làm mượt, không phải thời điểm bắt đầu lỗi."},
    {"id": "memory_not_persistence", "severity": "critical", "ids": ["alert.memory", "alert.prev_smoothed", "alert.hits", "F-EWMA-MEMORY"], "text": r"carried|memory|earlier bins|mang theo|bộ nhớ|các khoảng trước",
     "en": "This shows how the smoothing works, not that a condition persisted or began at a given time.",
     "vi": "Điều này cho thấy cách làm mượt hoạt động, không cho thấy một tình trạng đã kéo dài hay bắt đầu vào một thời điểm nhất định.",
     "short_en": "How the smoothing works, not a persisting condition.", "short_vi": "Cách làm mượt hoạt động, không phải tình trạng kéo dài."},
    {"id": "leak_demand", "severity": "critical", "ids": ["hyp.air_leakage", "hyp.downstream_demand", "F-IDENT", "unavailable:demand_measurement"], "text": r"\bleak|\bdemand|rò rỉ|nhu cầu",
     "en": "Leakage and higher downstream demand cannot be told apart with the data recorded for this asset.",
     "vi": "Với dữ liệu ghi nhận cho thiết bị này, không thể phân biệt rò rỉ với nhu cầu khí phía hạ nguồn tăng.",
     "short_en": "Leak and demand cannot be told apart with this data.", "short_vi": "Dữ liệu này không phân biệt được rò rỉ và nhu cầu."},
    {"id": "precursor", "severity": "critical", "ids": ["label.timing", "F-TIMING"], "text": r"precursor|\bearly\b|tiền thân|\bsớm\b",
     "en": "An alert preceded the documented onset; whether it was a precursor is not established.",
     "vi": "Một cảnh báo đã xuất hiện trước thời điểm bắt đầu được ghi nhận; chưa xác lập được đó có phải dấu hiệu báo trước hay không.",
     "short_en": "Not shown to be a precursor.", "short_vi": "Chưa xác lập là dấu hiệu báo trước."},
    {"id": "unmatched", "severity": "critical", "ids": ["bench.", "F-ALERT-BURDEN", "F-DISCRIMINATION"], "text": r"unmatched|benchmark-false|chưa gắn",
     "en": "Alerts not linked to a documented incident are not confirmed false alarms or confirmed healthy behaviour.",
     "vi": "Cảnh báo không gắn với sự cố đã ghi nhận không phải là báo động giả hay hoạt động khỏe mạnh đã được xác nhận.",
     "short_en": "Unmatched is not confirmed healthy or false.", "short_vi": "Chưa gắn không có nghĩa là khỏe mạnh hay báo giả."},
    {"id": "real_sim", "severity": "critical", "ids": ["bridge.status", "F-BRIDGE"], "text": r"kwh|energy|simulat|schedule|năng lượng|mô phỏng|lịch",
     "en": "Results from the real MetroPT-3 path are separate from simulation; no energy, flow or cost figure exists for this case.",
     "vi": "Kết quả từ dữ liệu MetroPT-3 thật tách biệt với mô phỏng; không có số liệu năng lượng, lưu lượng hay chi phí cho ca này.",
     "short_en": "Real evidence is separate from simulation.", "short_vi": "Bằng chứng thật tách biệt với mô phỏng."},
    {"id": "repair_scope", "severity": "context", "ids": ["cf.", "F-CF-NONUNIQUE"], "text": None,
     "en": "Only the five groups with the largest contributions were repaired, one at a time, and the alert was checked at the alert bin only.",
     "vi": "Chỉ năm nhóm tín hiệu đóng góp lớn nhất được sửa, mỗi lần một nhóm, và cảnh báo chỉ được kiểm tra tại khoảng thời gian cảnh báo."},
    {"id": "persistence_path", "severity": "context", "ids": ["F-CF-PERSIST", "detector.persistence"], "text": r"persistence|kiên trì|duy trì",
     "en": "A score above the threshold is not by itself an alert: the rule needs 3 of the last 4 smoothed scores at or above it.",
     "vi": "Điểm vượt ngưỡng chưa phải là cảnh báo: quy tắc cần 3 trong 4 điểm làm mượt gần nhất ở mức ngưỡng trở lên."},
    {"id": "units", "severity": "context", "ids": ["measured.", "F-TRAIN-RANGE"], "text": r"\bunits?\b|đơn vị|interquartile",
     "en": "Signal units are not loaded, and a value inside its training range is not shown to be normal.",
     "vi": "Đơn vị của tín hiệu chưa được nạp, và một giá trị nằm trong khoảng huấn luyện không có nghĩa là bình thường."},
    {"id": "calibration", "severity": "context", "ids": ["calrank.", "bench.selection", "F-DISCRIMINATION"], "text": r"calibration|hiệu chuẩn",
     "en": "Ranks compare against one calibration week in late February 2020, and the detectors were compared on the same TEST period.",
     "vi": "Thứ hạng so với một tuần hiệu chuẩn cuối tháng 2/2020, và các bộ phát hiện được so sánh trên cùng giai đoạn TEST."},
]
QUAL = {q["id"]: q for q in QUALS}
for _q in QUALS:
    _q["pattern"] = re.compile(_q["text"], re.IGNORECASE) if _q["text"] else None
PERSISTENCE_SAID = re.compile(r"(?<![\w.])(?:3|three) (?:of|out of) (?:the )?(?:last )?(?:4|four)(?!\d)|(?<![\w.])3 trong (?:số )?4(?!\d)|ba trong (?:số )?bốn", re.IGNORECASE)

# How to repair a failed check, sent with the one correction request. Fixed text; it names no record or finding.
FIX_HINTS = {
    "kind_mismatch": "Split that unit into separate units, one per record kind (for example the dataset label in one unit and the candidates in "
                     "another); records of kind unavailable may be cited alongside any unit.",
    "entity_not_cited": "Cite a record that carries that signal or candidate in the same unit, or leave the name out of that unit.",
    "number_not_cited": "Use only numbers and times shown in the records the unit cites, or cite the record that shows them.",
    "source_not_cited": "When a unit mentions documents, raw data or definitions, cite the matching unavailable record in that unit.",
    "required_finding": "Add a unit that states each named finding and cites it.",
    "not_offered": "Use only the purposes, records and findings listed in the schema.",
}

TEXT = {
    "operator_note": {"en": "Recorded as unverified for this session; it does not change any evidence or candidate status.",
                      "vi": "Đã ghi nhận là chưa kiểm chứng trong phiên này; điều đó không làm thay đổi bằng chứng hay trạng thái của các giả thuyết."},
    "also_relevant": {"en": "Also relevant", "vi": "Cũng liên quan (nguyên văn tiếng Anh)"},
    "failed": {"en": "I couldn't produce an answer that passes the evidence checks", "vi": "Tôi không tạo được câu trả lời vượt qua các kiểm tra bằng chứng"},
    "narrowed": {"en": "This question was too broad for one answer, so the evidence on {topics} was not used. Ask about it separately.",
                 "vi": "Câu hỏi quá rộng cho một câu trả lời, nên bằng chứng về {topics} chưa được dùng. Hãy hỏi riêng về phần đó."},
}
# Topic names shown to the user when a question had to be narrowed.
FOCUS_LABELS = {
    "alert": {"en": "why the alert fired", "vi": "vì sao cảnh báo xuất hiện"},
    "method": {"en": "how the numbers are computed", "vi": "cách tính các con số"},
    "timing": {"en": "how the evidence developed over the window", "vi": "diễn biến bằng chứng trong cửa sổ thời gian"},
    "attribution": {"en": "candidate explanations", "vi": "các giả thuyết giải thích"},
    "repair": {"en": "the detector repairs", "vi": "các phép sửa bộ phát hiện"},
    "label": {"en": "the documented incident", "vi": "sự cố đã ghi nhận"},
    "certainty": {"en": "whether the documented incident was a leak", "vi": "việc sự cố đã ghi nhận có phải là rò rỉ hay không"},
    "lead": {"en": "alert timing against the documented onset", "vi": "thời điểm cảnh báo so với lúc bắt đầu đã ghi nhận"},
    "signals": {"en": "the signal contributions", "vi": "đóng góp của các tín hiệu"},
    "range": {"en": "the signal values and training ranges", "vi": "giá trị tín hiệu và khoảng huấn luyện"},
    "benchmark": {"en": "the benchmark results", "vi": "kết quả đánh giá"},
    "energy": {"en": "energy and simulation", "vi": "năng lượng và mô phỏng"},
    "sources": {"en": "the sources", "vi": "các nguồn"},
}


def triggers(q: dict, direct: set[str], text: str) -> bool:
    """Whether a qualification applies to a unit that cites `direct` record and finding IDs and says `text`."""
    return any(i in direct if not i.endswith(".") else any(d.startswith(i) for d in direct) for i in q["ids"]) or bool(q["pattern"] and q["pattern"].search(text))


def depth_after(depth: str, minimum: str) -> bool:
    return DEPTHS.index(depth) >= DEPTHS.index(minimum)


# Question focus -> what a turn about it needs. `find` are findings (their input records come with them), `brief` findings that come
# with their LEAN anchors only, `rec` extra record IDs
# (a prefix when it ends with "." or ":"), `purposes` the purposes beyond CORE_PURPOSES an answer may declare. A focus is matched
# from the question text in English and Vietnamese. Order is priority: when a prompt does not fit the token budget the
# last-listed matched focuses are dropped first, whole.
# Every turn also carries a record each refusal can cite (REFUSALS), so a refusal can always be validated, and the data-gap finding.
CORE_RECORDS = ["alert.time", "alert.threshold", "alert.smoothed", "alert.margin", "unavailable:",
                *dict.fromkeys(c for r in REFUSALS.values() for c in r["cite"] if not c.startswith("F-"))]
CORE_FINDINGS = ["F-GAP"]
# A finding that is in a turn without the question being about it (a purpose requires it, or a focus lists it as `brief`) comes with
# just these records to cite beside it instead of all its inputs. Its text still carries every figure it states.
LEAN = {"F-EVIDENCE-TIMING": ["temporal.window"], "F-CF-NONUNIQUE": ["alert.threshold"]}
CORE_PURPOSES = ["evidence_and_sources", "outside_evidence"]
DEFAULT_FOCUS = ["alert"]  # a question that matches nothing and follows no earlier topic
FOCUS_PATTERNS = {
    "alert": r"\bfir(?:e|ed|es|ing)\b|\btrigger|\bthreshold|\bmargin\b|\bscores?\b|\bsmooth|\bewma\b|\bcarried\b|\bmemory\b|\bpersist"
             r"|cảnh báo[^.?!]{0,20}(?:xuất hiện|phát ra|kêu)|kích hoạt|ngưỡng|điểm|làm mượt|bộ nhớ",
    "method": r"\bformula|\bequation|\bcalculat|\bcomput|\bderiv|\balpha\b|\bpca\b|\breconstruction\b|\bwhere does\b|\bcome from\b|\bhow (?:is|are)\b|\bmethod"
              r"|công thức|\btính\b|từ đâu|phương pháp",
    "timing": r"\b\d{1,2}:\d{2}\b|\bjumps?\b|\bros[e]\b|\brises?\b|\brising\b|\bspik|\bwhen\b|\bemerg|\bover time\b|\btrend|\bpercentiles?\b|\bwindow\b"
              r"|khi nào|\blúc\b|\btăng\b|nhảy|phân vị|cửa sổ",
    "attribution": r"\bcaus|\bresponsib|\bdue to\b|\bbecause\b|\bdriv(?:e|es|en|ing)\b|\bbehind\b|\bleak|\bdemand|\bregulat|\bdryer|\bdiagnos|\bfault|\bbroke|\bbroken"
                   r"|\bfail|\bproblem|\bhypothes|\bcandidates?\b|\breason|\bdefinitely\b|\bcertain|\bsure\b|\bwrong\b|\btechnician|\boperators?\b|\bignore\b|\bsafe\b|\bnormal\b"
                   r"|rò rỉ|nhu cầu|nguyên nhân|chẩn đoán|\blỗi\b|hỏng|giả thuyết|điều áp|chắc|kỹ thuật viên|vận hành viên|bỏ qua|an toàn|bình thường",
    "repair": r"\brepair|\bremov|\bfix|\breplac|\breconstruct|\bcounterfactual|\bclears?\b|\bcleared\b|\bwhat if\b|\bwithout\b|sửa|thay thế|loại bỏ|\bgỡ\b|\bxóa\b|tái tạo",
    "label": r"\blabel|\bdocumented\b|\bincident|\bfailure list|\bprovider\b|ground truth|\bair dryer\b|\b20(?:19|2\d)\b|nhãn|sự cố|máy sấy",
    # Onset, prediction and lead time: what the alert's timing against the documented onset does and does not show.
    "lead": r"\bonset\b|\bprecursor|\bpredict|\bforecast|\bearly\b|\bahead\b|in advance|lead time|before the (?:incident|failure|leak)|dự đoán|dự báo|\bsớm\b|trước khi",
    # A short question about whether the documented incident was definitely a leak (or another named fault). A question that also asks
    # about causes, signals, repairs or PCA evidence is an attribution question instead.
    "certainty": r"(?s)\A(?!.*(?:\bcaus|\bresponsib|\bdriv|\bsignal|\bcontribut|\brepair|\bpca\b|\bregulat|\bsensor|\bcompressor|nguyên nhân|gây ra|tín hiệu|đóng góp|sửa|điều áp|cảm biến))"
                 r"(?=.*(?:\bdefinitely\b|\bcertain|\bsure\b|\bconfirm|\bactually\b|\breally\b|\bproven?\b|chắc chắn|xác nhận|thật sự|có phải))"
                 r"(?=.*(?:\bleak|\bdryer|\bfault|rò rỉ|máy sấy|\blỗi\b))",
    "signals": r"\btp[23]\b|\bh1\b|dv[_ ]?pressure|dv[_ ]?electric|motor[_ ]?current|oil[_ ]?(?:temperature|level)|reservoirs?|\bcomp\b|\btowers?\b|\bmpg\b|\blps\b"
               r"|pressure[_ ]?switch|caudal|\bsignals?\b|\bgroups?\b|\bcontribut|\bshares?\b|\brank|tín hiệu|nhóm|đóng góp|tỷ trọng|xếp hạng|áp suất",
    "range": r"\brange|\bunits?\b|interquartile|\btraining\b|\bfeatures?\b|đơn vị|khoảng huấn luyện|đặc trưng|huấn luyện",
    "benchmark": r"false alarms?|false positives?|\bbenchmark|\bprecision|\brecall|pr-?auc|\bepisodes?\b|\bburden|how often|\bper (?:day|24)|\bexposure|test period"
                 r"|\bselect|\bhealthy\b|\breliab|\baccura|how many|báo (?:động )?giả|độ chính xác|thường xuyên|khỏe|bao nhiêu lần",
    "energy": r"energy|\bkwh\b|\bcosts?\b|\bsaving|\bsaved\b|\bwast|\bpower\b|\bflow\b|\bschedul|\bsimulat|\bbridge\b|\bapply\b|\belectricity\b|\bmoney|\bbill"
              r"|năng lượng|điện|chi phí|tiết kiệm|lãng phí|\blịch\b|mô phỏng|áp dụng|công suất|lưu lượng",
    "sources": r"\bmanuals?\b|\boem\b|\bsops?\b|datasheet|maintenance|\blogs?\b|\btickets?\b|\braw\b|\btraces?\b|telemetry|waveform|\bseries\b|\bdefin(?:e|ed|es|itions?)\b|\bwhat is\b"
               r"|\bwhat are\b|\bmeaning\b|stands for|\bshow me\b|\bdocuments?\b|sổ tay|nhật ký|bảo trì|\bthô\b|đồ thị|định nghĩa|là gì|cho tôi xem|tài liệu",
}
FOCUS = {
    "alert": {"find": ["F-MARGIN", "F-EWMA-MEMORY"], "rec": ["contrib.", "calrank.", "detector.persistence", "alert.hits"], "purposes": ["what_happened"]},
    "method": {"find": ["F-EWMA-MEMORY"], "rec": ["detector.ewma", "detector.persistence", "detector.calibration_bins", "detector.repair", "alert.hits"],
               "purposes": ["method"]},
    "timing": {"find": ["F-EVIDENCE-TIMING"], "rec": ["contrib."], "purposes": []},
    "certainty": {"find": ["F-LABEL", "F-IDENT"], "rec": [], "purposes": []},
    "attribution": {"find": ["F-CANDIDATES", "F-IDENT", "F-LABEL", "F-TIMING"], "brief": ["F-CF-NONUNIQUE"], "rec": [], "purposes": ["possible_causes"]},
    "repair": {"find": ["F-CF-NONUNIQUE", "F-CF-PERSIST"], "rec": ["detector.repair", "detector.persistence", "contrib."], "purposes": []},
    "lead": {"find": ["F-TIMING", "F-LABEL"], "rec": [], "purposes": []},
    "label": {"find": ["F-LABEL", "F-TIMING", "F-SELECTION"], "rec": [], "purposes": []},
    "signals": {"find": [], "rec": ["contrib.", "calrank."], "purposes": []},
    "range": {"find": ["F-TRAIN-RANGE"], "rec": [], "purposes": []},
    "benchmark": {"find": ["F-ALERT-BURDEN", "F-DISCRIMINATION", "F-SELECTION"], "rec": [], "purposes": []},
    "energy": {"find": ["F-BRIDGE"], "rec": [], "purposes": []},
    "sources": {"find": [], "rec": [], "purposes": []},
}
# What an answer on a topic must convey, set by the server from the topic and the depth, not by the purposes the model declares:
# (finding, shallowest depth at which it is required). A finding that does not fire for the case is never required. The model's
# declared purposes can add requirements (PURPOSES) but never remove these. Topics that were dropped to fit the budget add none.
ESSENTIAL = {
    "alert": [("F-MARGIN", "quick"), ("F-EWMA-MEMORY", "quick"), ("F-EVIDENCE-TIMING", "investigation")],
    "timing": [("F-EVIDENCE-TIMING", "quick")],
    "attribution": [("F-CANDIDATES", "quick"), ("F-IDENT", "quick"), ("F-CF-NONUNIQUE", "investigation")],
    "repair": [("F-CF-NONUNIQUE", "quick"), ("F-CF-PERSIST", "quick")],
    "certainty": [("F-LABEL", "quick"), ("F-IDENT", "quick")],
    "lead": [("F-TIMING", "quick")],
    "label": [("F-LABEL", "quick")],
    "benchmark": [("F-ALERT-BURDEN", "quick")],
    "range": [("F-TRAIN-RANGE", "quick")],
    "energy": [("F-BRIDGE", "quick")],
}
# A narrower topic subsumes the broader one when both match: a short "was it definitely a leak?" asks for the label and the
# identifiability limit, not for the candidate list or the repair evidence, and must not be offered the broader topic's purposes.
NARROWS = {"certainty": "attribution"}
# What counts as conveying a finding that states no figure to repeat: every group must match the text of the units that cite it. A
# matching timestamp or number is not enough. Draft Vietnamese wording; it needs the owner's review.
CONVEY = {
    # the label as transcribed from the provider's list; that AeroXAI's evidence does not establish it
    "F-LABEL": [
        r"air[- ]leak|high[- ]stress|rò rỉ",
        r"provider|failure list|transcri|repository|configuration|nhà cung cấp|danh sách|phiên âm|sao chép|cấu hình|kho mã",
        (r"\bnot (?:an? |the )?(?:independent|established|confirmed|verified|proof)|n't (?:an? )?(?:independent|establish|confirm|verif)"
         r"|\bnot (?:establish|confirm|verif)|no independent|cannot (?:establish|confirm|verify)"
         r"|không (?:xác lập|xác nhận|chứng minh|kiểm chứng|độc lập|thiết lập)|chưa (?:xác lập|xác nhận|được kiểm chứng|kiểm chứng)"),
    ],
    # leakage and downstream demand, and that this data cannot separate them
    "F-IDENT": [
        r"leak|rò rỉ",
        r"demand|nhu cầu",
        (r"cannot (?:be )?(?:told apart|distinguished|separated)|cannot (?:tell|distinguish|separate)|can't (?:tell|distinguish|separate)"
         r"|(?:not|n't) (?:be )?(?:able to )?(?:tell|distinguish|separate)|indistinguishable|không (?:thể )?(?:phân biệt|tách|phân định)"),
    ],
}
CONVEY = {f: [re.compile(rx, re.IGNORECASE) for rx in groups] for f, groups in CONVEY.items()}
# Words that open a question without saying what it is about; they never pick a topic, so "Why did that happen?" follows the last one.
GENERIC = re.compile(r"\bwhy\b|\bhow come\b|vì sao|tại sao|sao lại", re.IGNORECASE)
for _name, _rx in FOCUS_PATTERNS.items():
    FOCUS[_name]["pattern"] = re.compile(_rx, re.IGNORECASE)

# A qualification is not repeated beside a unit whose own text already states what it says: all patterns must match that text.
QUAL["leak_demand"]["said"] = CONVEY["F-IDENT"]
QUAL["persistence_path"]["said"] = [PERSISTENCE_SAID]
