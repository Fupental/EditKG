# -*- coding: utf-8 -*-
"""
relation_templates.py — 全部39个关系 → 自然语言模板的映射
模板中 {head} 和 {tail} 分别代表头实体和尾实体的可读名称。

说明：
1. 对于"类型/本体关系"，虽然不建议作为最终书籍属性知识使用，但仍提供描述性模板，
   便于调试、审计或中间结果可视化。
2. 对于"日期关系"，当前 tail 往往不是实际日期，而是 XMLSchema 类型标记，
   因此模板写成"date value recorded as datatype ..."，避免误导。
3. 对于 Freebase 内部元数据关系，也统一补成元数据表达，建议下游默认过滤。

使用方式：
    from utils.relation_templates import REL_TEMPLATES
    stmt = REL_TEMPLATES['book.written_work.author'].format(head='...', tail='...')
"""

REL_TEMPLATES = {

    # ============================================================
    # 一、类型/本体关系（共 ~2,211,636 条三元组）
    # ============================================================

    # 666,473条 | 例: ("book.written_work", type.type.instance, "Lion's Blood") — 类型→实例
    'type.type.instance':
        '"{tail}" is an instance of the type "{head}"',

    # 665,931条 | 例: ("m.06d45vr", type.object.type, "book.written_work") — 实例→类型
    'type.object.type':
        '"{head}" is of type "{tail}"',

    # 665,931条 | 例: ("South Sea Tales", 22-rdf-syntax-ns#type, "base.type_ontology.inanimate") — RDF类型声明
    '22-rdf-syntax-ns#type':
        '"{head}" has RDF type "{tail}"',

    # 106,664条 | 例: ("m.06pw_zy", kg.object_profile.prominent_type, "book.book") — 实体的主要类型
    'kg.object_profile.prominent_type':
        'the prominent type of "{head}" is "{tail}"',

    # 106,637条 | 例: ("m.06rqybl", common.topic.notable_types, "m.01xryvm") — 实体的显著类型
    'common.topic.notable_types':
        '"{head}" is notably associated with the type "{tail}"',

    # ============================================================
    # 二、书籍核心关系（共 ~304,000 条三元组）
    # ============================================================

    # 50,428条 | 例: ("Genre Fiction", → , "Infernal: A Repairman Jack Novel") — 类型包含某书
    'media_common.literary_genre.books_in_this_genre':
        '"{head}" is a genre that includes the book "{tail}"',

    # 50,276条 | 例: ("Lord Valentine's Castle", → , "Genre Fiction") — 书属于某类型
    'book.book.genre':
        '"{head}" belongs to the genre "{tail}"',

    # 40,193条 | 例: ("Wolves Eat Dogs", → , "Religion & Spirituality") — 书的主题
    'book.written_work.subjects':
        '"{tail}" is the subject of "{head}"',

    # 40,126条 | 例: ("Mystery, Thriller & Suspense", → , "No Comebacks") — 主题包含某书
    'book.book_subject.works':
        '"{head}" is the subject of "{tail}"',

    # 37,426条 | 例: ("The Birchbark House", → , "Louise Erdrich") — 书的作者
    'book.written_work.author':
        '"{head}" was written by "{tail}"',

    # 37,351条 | 例: ("Terry Jones", → , "Lady Cottington's Fairy Album") — 作者写了某书
    'book.author.works_written':
        '"{head}" wrote "{tail}"',

    # 13,923条 | 例: ("The Secret Panel", → , "English") — 书的原始语言
    'book.written_work.original_language':
        '"{head}" was originally written in "{tail}"',

    # 10,182条 | 例: ("Nancy Drew", → , "The Secret of the Old Clock") — 角色出现在某书
    'book.book_character.appears_in_book':
        '"{head}" appears in the book "{tail}"',

    # 10,062条 | 例: ("Queen of the Amazons", → , "Alexander") — 书包含某角色
    'book.book.characters':
        '"{head}" features the character "{tail}"',

    # 3,880条 | 例: ("Meredith Gentry", → , "Divine Misdemeanors") — 系列包含某书
    'book.literary_series.works_in_this_series':
        '"{head}" is a series containing "{tail}"',

    # 3,856条 | 例: ("A Lick of Frost", → , "Meredith Gentry") — 书属于某系列
    'book.written_work.part_of_series':
        '"{head}" is part of the series "{tail}"',

    # 2,397条 | 例: ("Lincoln Unmasked", → , "The Real Lincoln") — 书的前一本
    'book.written_work.previous_in_series':
        '"{head}" is preceded by "{tail}" in its series',

    # 2,394条 | 例: ("The Lost Boy", → , "A Man Named Dave") — 书的后一本
    'book.written_work.next_in_series':
        '"{head}" is followed by "{tail}" in its series',

    # 164条 | 例: ("The Ugly Little Boy", → , "Science Fiction & Fantasy") — 短篇的类型
    'book.short_story.genre':
        '"{head}" belongs to the genre "{tail}"',

    # 164条 | 例: ("Short Stories & Anthologies", → , "The Moon Moth") — 类型包含某短篇
    'media_common.literary_genre.stories_in_this_genre':
        '"{head}" is a genre that includes the story "{tail}"',

    # 162条 | 例: ("Edward Gorey", → , "The House with a Clock in Its Walls") — 插画师画了某书
    'book.illustrator.books_illustrated':
        '"{head}" illustrated "{tail}"',

    # 161条 | 例: ("Joseph Had a Little Overcoat", → , "Simms Taback") — 书的插画师
    'book.book.interior_illustrations_by':
        '"{head}" was illustrated by "{tail}"',

    # ============================================================
    # 三、戏剧关系
    # ============================================================

    # 317条 | 例: ("The Glass Menagerie", → , "United States") — 戏剧的起源国
    'theater.play.country_of_origin':
        '"{head}" originated in "{tail}"',

    # 280条 | 例: ("700 Sundays", → , "Comedy") — 戏剧的类型
    'theater.play.genre':
        '"{head}" belongs to the theatrical genre "{tail}"',

    # 280条 | 例: ("Comedy", → , "Man of La Mancha") — 戏剧类型包含某剧
    'theater.theater_genre.plays_in_this_genre':
        '"{head}" is a genre that includes the play "{tail}"',

    # ============================================================
    # 四、虚构世界关系
    # ============================================================

    # 303条 | 例: ("Doctor Who Universe", → , "The Last Dodo") — 虚构世界包含某作品
    'fictional_universe.fictional_universe.works_set_here':
        '"{tail}" is set in the fictional universe "{head}"',

    # 301条 | 例: ("The Last Dodo", → , "Doctor Who Universe") — 作品设定在某虚构世界
    'fictional_universe.work_of_fiction.part_of_these_fictional_universes':
        '"{head}" is set in the fictional universe "{tail}"',

    # ============================================================
    # 五、日期关系（尾实体是 XMLSchema 日期类型标记，不是实际日期值）
    # ============================================================

    # 19,706条 | 例: ("m.06gld0p", → , "XMLSchema#gYear") — 首次出版日期
    'book.written_work.date_of_first_publication':
        '"{head}" has a recorded date of first publication with datatype "{tail}"',

    # 11,700条 | 例: ("m.04t1rw_", → , "XMLSchema#gYear") — 版权日期
    'book.written_work.copyright_date':
        '"{head}" has a recorded copyright date with datatype "{tail}"',

    # 320条 | 例: ("The History Boys", → , "XMLSchema#date") — 戏剧首演日期
    'theater.play.date_of_first_performance':
        '"{head}" has a recorded date of first performance with datatype "{tail}"',

    # 235条 | 例: ("m.02gfxv6", → , "XMLSchema#gYear") — 写作日期
    'book.written_work.date_written':
        '"{head}" has a recorded writing date with datatype "{tail}"',

    # ============================================================
    # 六、Freebase 内部元数据关系
    # ============================================================

    # 4,752条 | 例: ("m.045rjpk", → , "m.03jz7pj") — Freebase审核标记
    'freebase.valuenotation.is_reviewed':
        '"{head}" has the Freebase review marker "{tail}"',

    # 2,862条 | 例: ("m.0268fzp", → , "m.03jz7nm") — Freebase有值标记
    'freebase.valuenotation.has_value':
        '"{head}" has the Freebase value-present marker "{tail}"',

    # 477条 | 例: ("m.0y56549", → , "m.04s_yz5") — Freebase无值标记
    'freebase.valuenotation.has_no_value':
        '"{head}" has the Freebase no-value marker "{tail}"',

    # ============================================================
    # 七、标签/名称/链接关系
    # ============================================================

    # 362条 | 例: ("m.06jhy7r", → , '"I Love You"') — RDF标签
    'rdf-schema#label':
        'the RDF label of "{head}" is "{tail}"',

    # 362条 | 例: ("Hello, Gorgeous!", → , '"Hello"@en') — 实体名称
    'type.object.name':
        'the name of "{head}" is "{tail}"',

    # 350条 | 例: ("Thinking In Numbers", → , URL实体) — 官方网站
    'common.topic.official_website':
        'the official website of "{head}" is "{tail}"',

    # 223条 | 例: ("m.042zk64", → , URL实体) — 相关网页
    'common.topic.topical_webpage':
        '"{head}" has the topical webpage "{tail}"',

    # 135条 | 例: ("m.06h_8hr", → , "m.0266zsb") — Freebase用户话题标记
    'base.yupgrade.user.topics':
        '"{head}" is associated with the user topic marker "{tail}"',
}
