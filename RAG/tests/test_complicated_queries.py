"""
test_complicated_queries.py

Benchmark test set of 25 complicated Egyptian civil law questions
with known correct article references.

These queries are designed to test multi-article reasoning,
cross-concept retrieval, and nuanced legal understanding --
the exact scenarios where the RAG pipeline was failing.

Usage:
    Run as a standalone script to validate retrieval quality:
        python -m pytest RAG/tests/test_complicated_queries.py -v

    Or import BENCHMARK_QUERIES for use in evaluation scripts.
"""

# Each entry contains:
#   - query: The complicated legal question in Arabic
#   - expected_articles: List of article numbers that should be retrieved
#   - category: The type of complexity involved
#   - difficulty: easy | medium | hard

BENCHMARK_QUERIES = [
    # --- Multi-article reasoning (cross-article dependencies) ---
    {
        "query": "ما هي شروط صحة العقد وما هي حالات البطلان المطلق والنسبي؟",
        "expected_articles": [89, 90, 91, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
        "category": "multi_article",
        "difficulty": "hard",
    },
    {
        "query": "كيف ينظم القانون المدني المصري المسؤولية التقصيرية وما علاقتها بالتعويض؟",
        "expected_articles": [163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222],
        "category": "multi_article",
        "difficulty": "hard",
    },
    {
        "query": "ما هي أحكام الإيجار من حيث التزامات المؤجر والمستأجر وأسباب انتهاء العقد؟",
        "expected_articles": [558, 559, 560, 561, 562, 563, 564, 566, 567, 568, 569, 598, 599, 600, 601],
        "category": "multi_article",
        "difficulty": "hard",
    },
    {
        "query": "ما الفرق بين الفسخ والانفساخ في العقود الملزمة للجانبين؟",
        "expected_articles": [157, 158, 159, 160],
        "category": "multi_article",
        "difficulty": "medium",
    },
    {
        "query": "ما هي شروط المقاصة القانونية وكيف تختلف عن المقاصة الاتفاقية؟",
        "expected_articles": [362, 363, 364, 365, 366, 367],
        "category": "multi_article",
        "difficulty": "medium",
    },

    # --- Nuanced legal concept queries ---
    {
        "query": "هل يجوز للقاضي تعديل الشرط الجزائي إذا أثبت المدين أن التقدير مبالغ فيه؟",
        "expected_articles": [223, 224],
        "category": "nuanced_concept",
        "difficulty": "medium",
    },
    {
        "query": "ما هو أثر الغلط في القانون على صحة العقد مقارنة بالغلط في الواقع؟",
        "expected_articles": [120, 121, 122, 123, 124],
        "category": "nuanced_concept",
        "difficulty": "hard",
    },
    {
        "query": "متى يعتبر السكوت قبولاً في التعاقد وفقاً للقانون المدني المصري؟",
        "expected_articles": [98, 99],
        "category": "nuanced_concept",
        "difficulty": "medium",
    },
    {
        "query": "ما مدى مسؤولية المتبوع عن أعمال التابع غير المشروعة وهل يحق له الرجوع على التابع؟",
        "expected_articles": [174, 175],
        "category": "nuanced_concept",
        "difficulty": "medium",
    },
    {
        "query": "كيف يتعامل القانون مع حالة استحالة التنفيذ بسبب قوة قاهرة في العقود الملزمة للجانبين؟",
        "expected_articles": [159, 165, 215, 373],
        "category": "nuanced_concept",
        "difficulty": "hard",
    },

    # --- Cross-chapter reasoning ---
    {
        "query": "ما العلاقة بين أحكام التقادم المسقط والتقادم المكسب في القانون المدني المصري؟",
        "expected_articles": [374, 375, 376, 377, 378, 968, 969, 970],
        "category": "cross_chapter",
        "difficulty": "hard",
    },
    {
        "query": "كيف يؤثر تسجيل العقد في نقل الملكية العقارية وما علاقته بأحكام البيع؟",
        "expected_articles": [418, 419, 420, 934, 935, 936],
        "category": "cross_chapter",
        "difficulty": "hard",
    },
    {
        "query": "ما هي العلاقة بين الحيازة والتقادم المكسب وكيف تؤثر الحيازة في كسب الملكية؟",
        "expected_articles": [949, 950, 951, 968, 969],
        "category": "cross_chapter",
        "difficulty": "hard",
    },

    # --- Procedural / applied queries ---
    {
        "query": "إذا باع شخص عقاراً مملوكاً للغير فما هي حقوق المشتري وما هو مصير العقد؟",
        "expected_articles": [466, 467, 468],
        "category": "applied",
        "difficulty": "medium",
    },
    {
        "query": "ما هي شروط دعوى الإثراء بلا سبب ومتى يلتزم المثري بالتعويض؟",
        "expected_articles": [179, 180, 181, 182],
        "category": "applied",
        "difficulty": "medium",
    },
    {
        "query": "كيف يحسب التعويض عن الضرر الأدبي وهل يورث الحق في المطالبة به؟",
        "expected_articles": [222, 223],
        "category": "applied",
        "difficulty": "medium",
    },
    {
        "query": "ما هي أحكام الكفالة الشخصية وما الفرق بينها وبين الكفالة العينية؟",
        "expected_articles": [772, 773, 774, 775, 776, 777],
        "category": "applied",
        "difficulty": "medium",
    },
    {
        "query": "ما هو حكم التصرف في مال المستقبل وهل يجوز بيع شيء مستقبلي؟",
        "expected_articles": [131, 418, 419],
        "category": "applied",
        "difficulty": "medium",
    },

    # --- Ambiguous / vague queries that need refinement ---
    {
        "query": "ما هو حكم القانون في التعامل مع الجار؟",
        "expected_articles": [807, 808, 809, 810, 811],
        "category": "ambiguous",
        "difficulty": "easy",
    },
    {
        "query": "ما حكم الضمان في القانون المدني؟",
        "expected_articles": [443, 444, 445, 446, 447],
        "category": "ambiguous",
        "difficulty": "easy",
    },

    # --- Edge cases ---
    {
        "query": "هل يجوز الجمع بين المسؤولية العقدية والتقصيرية في دعوى واحدة؟",
        "expected_articles": [163, 215, 221],
        "category": "edge_case",
        "difficulty": "hard",
    },
    {
        "query": "ما هو أثر وفاة أحد المتعاقدين على العقد وهل ينتقل الالتزام للورثة؟",
        "expected_articles": [145, 146],
        "category": "edge_case",
        "difficulty": "medium",
    },
    {
        "query": "كيف يتعامل القانون مع تعدد المسؤولين عن فعل ضار واحد وما هي قواعد التضامن بينهم؟",
        "expected_articles": [169, 170],
        "category": "edge_case",
        "difficulty": "medium",
    },
    {
        "query": "ما الحكم إذا تعارض شرط في العقد مع نص آمر في القانون المدني؟",
        "expected_articles": [131, 132, 133],
        "category": "edge_case",
        "difficulty": "medium",
    },
    {
        "query": "هل يجوز الاتفاق على الإعفاء من المسؤولية عن الفعل العمد أو الخطأ الجسيم؟",
        "expected_articles": [217, 218],
        "category": "edge_case",
        "difficulty": "medium",
    },
]


def test_benchmark_queries_exist():
    """Verify the benchmark set has the expected number of queries."""
    assert len(BENCHMARK_QUERIES) == 25


def test_all_queries_have_required_fields():
    """Every benchmark query must have query, expected_articles, category, difficulty."""
    required_keys = {"query", "expected_articles", "category", "difficulty"}
    for i, q in enumerate(BENCHMARK_QUERIES):
        missing = required_keys - set(q.keys())
        assert not missing, f"Query {i} missing keys: {missing}"


def test_all_queries_have_nonempty_articles():
    """Each query should reference at least one expected article."""
    for i, q in enumerate(BENCHMARK_QUERIES):
        assert len(q["expected_articles"]) > 0, f"Query {i} has no expected articles"


def test_difficulty_values():
    """Difficulty must be one of easy, medium, hard."""
    valid = {"easy", "medium", "hard"}
    for i, q in enumerate(BENCHMARK_QUERIES):
        assert q["difficulty"] in valid, f"Query {i} has invalid difficulty: {q['difficulty']}"


def test_category_values():
    """Category must be one of the defined types."""
    valid = {"multi_article", "nuanced_concept", "cross_chapter", "applied", "ambiguous", "edge_case"}
    for i, q in enumerate(BENCHMARK_QUERIES):
        assert q["category"] in valid, f"Query {i} has invalid category: {q['category']}"
