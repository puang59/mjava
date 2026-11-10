from mjava.pipeline.mjava_file import COMMENT, JAVA, SHORTHAND, assemble, parse_mjava

SLIDE = (
    "// Custom syntax a user would actually write --\n"
    "// natural language instructions, not new syntax rules\n"
    "Create a class Student with fields: name as String,\n"
    "  age as int, gpa as double.\n"
)


def kinds(text):
    return [b.kind for b in parse_mjava(text)]


def test_slide_example():
    bl = parse_mjava(SLIDE)
    assert [b.kind for b in bl] == [COMMENT, SHORTHAND]
    assert bl[0].attached
    assert bl[1].instruction == "Create a class Student with fields: name as String, age as int, gpa as double."
    assert (bl[1].start_line, bl[1].end_line) == (3, 4)


def test_empty_and_whitespace_only():
    assert parse_mjava("") == []
    assert parse_mjava("\n\n  \n") == []


def test_no_shorthand_all_java():
    t = "import java.util.List;\n\nclass A {\n    int x;\n}\n"
    assert kinds(t) == [JAVA, JAVA]


def test_java_passthrough_variants():
    for java in [
        "package a.b;",
        "import java.util.*;",
        "import static java.lang.Math.max;",
        "public class A { }",
        "@FunctionalInterface\ninterface F { void f(); }",
        "enum Color { RED, GREEN }",
        "record P(int x, int y) {}",
        "int x = 5;",
        "foo(1, 2);",
        "static final int MAX = 3;",
    ]:
        assert kinds(java) == [JAVA], java


def test_sentences_are_shorthand():
    for s in [
        "Create a class A with field x as int.",
        "class Student with name and age",  # keyword start, but no brace/semicolon
        "public class Foo with a name field",
        "Class Student has fields name; age;",  # ends in ; but no Java characters
        "an enum Color with RED, GREEN",
    ]:
        assert kinds(s) == [SHORTHAND], s


def test_multiline_instruction_joined():
    bl = parse_mjava("Create a class A\n   with field x\n  as int.")
    assert len(bl) == 1 and bl[0].instruction == "Create a class A with field x as int."


def test_blank_line_splits_instructions():
    bl = parse_mjava("Create a class A.\n\nCreate a class B.")
    assert [b.instruction for b in bl] == ["Create a class A.", "Create a class B."]


def test_gen_marker_forces_shorthand():
    bl = parse_mjava("@gen int x = 5;")
    assert bl[0].kind == SHORTHAND and bl[0].instruction == "int x = 5;"
    bl = parse_mjava("@gen\npublic class A { }\nwith a field")
    assert bl[0].kind == SHORTHAND and bl[0].instruction == "public class A { } with a field"
    bl = parse_mjava("@gen: make a counter class")
    assert bl[0].instruction == "make a counter class"


def test_annotation_is_java_but_gen_is_not():
    assert kinds("@Override\npublic String toString() { return \"\"; }") == [JAVA]
    assert kinds("@gen a thing") == [SHORTHAND]


def test_blank_lines_inside_java_class_do_not_split():
    t = "class A {\n    int x;\n\n    int f() {\n\n        return x;\n    }\n}\n\nMake a class B."
    bl = parse_mjava(t)
    assert [b.kind for b in bl] == [JAVA, SHORTHAND]
    assert "int f()" in bl[0].text and "\n\n" in bl[0].text


def test_braces_in_strings_do_not_confuse():
    t = 'class A {\n  String s = "{";\n\n  int y;\n}\n'
    assert kinds(t) == [JAVA]


def test_block_comment_with_blank_line_stays_together():
    t = "/* first\n\n second */\nMake a class A."
    bl = parse_mjava(t)
    assert [b.kind for b in bl] == [COMMENT, SHORTHAND]
    assert "second" in bl[0].text


def test_comment_only_and_comment_between_blocks():
    t = "// just a note\n\nMake a class A.\n\n/** doc */\nclass B { }"
    bl = parse_mjava(t)
    assert [b.kind for b in bl] == [COMMENT, SHORTHAND, COMMENT, JAVA]
    assert not bl[0].attached and bl[2].attached


def test_comment_lines_inside_shorthand_are_not_instruction():
    bl = parse_mjava("Make a class A\n// hidden note\nwith field x.")
    assert bl[0].instruction == "Make a class A with field x."


def test_windows_line_endings():
    t = SLIDE.replace("\n", "\r\n") + "\r\nclass B { }\r\n"
    bl = parse_mjava(t)
    assert [b.kind for b in bl] == [COMMENT, SHORTHAND, JAVA]
    assert "\r" not in bl[1].instruction


def test_line_numbers_are_one_based_and_accurate():
    t = "// c\n\nclass A { }\n\n\nMake a class B\nwith x."
    bl = parse_mjava(t)
    assert [(b.start_line, b.end_line) for b in bl] == [(1, 1), (3, 3), (6, 7)]


def test_assemble_hoists_dedups_and_keeps_package_first():
    t = (
        "import java.util.List;\n\nMake A.\n\nclass Keep { List<String> l; }\n\n"
        "package demo;\n\nMake B."
    )
    bl = parse_mjava(t)
    sh = [b for b in bl if b.kind == SHORTHAND]
    gen = {
        sh[0].index: "import java.util.List;\nimport java.util.Map;\nclass A { }",
        sh[1].index: "package other;\nimport java.util.Map;\nclass B { }",
    }
    asm = assemble(bl, gen)
    lines = asm.text.split("\n")
    assert lines[0] == "package demo;"  # first package wins, moved to the top
    assert asm.text.count("import java.util.List;") == 1
    assert asm.text.count("import java.util.Map;") == 1
    assert "package other" not in asm.text
    last_import = max(i for i, l in enumerate(lines) if l.startswith("import"))
    first_decl = min(i for i, l in enumerate(lines) if l.startswith("class"))
    assert last_import < first_decl
    assert asm.text.endswith("\n")


def test_assemble_maps_lines_back_to_blocks():
    bl = parse_mjava("class Keep { }\n\nMake A.\n\nMake B.")
    sh = [b for b in bl if b.kind == SHORTHAND]
    asm = assemble(bl, {sh[0].index: "class A {\n    int x;\n}", sh[1].index: "import java.util.List;\nclass B {\n  int y;\n}"})
    lines = asm.text.split("\n")
    ln = lines.index("  int y;") + 1
    assert asm.locate(ln) == (sh[1].index, 3)  # third line of B's own code
    imp = lines.index("import java.util.List;") + 1
    assert asm.locate(imp) == (sh[1].index, 1)  # hoisted import maps to its raw line
    assert asm.locate(lines.index("class Keep { }") + 1)[0] == bl[0].index
    assert asm.locate(9999) == (None, -1)


def test_assemble_omits_missing_shorthand():
    bl = parse_mjava("class K { }\n\nMake A.")
    assert assemble(bl, {}).text == "class K { }\n"
    assert assemble([], {}).text == ""


def test_package_only_in_literal_stays_first_with_comments():
    bl = parse_mjava("// header\n\npackage p;\n\nclass A { }")
    lines = assemble(bl, {}).text.split("\n")
    assert lines[0] == "// header"  # leading file comments stay on top
    assert [l for l in lines if l.strip() and not l.startswith("//")][0] == "package p;"
