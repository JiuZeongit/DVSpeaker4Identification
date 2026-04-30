import textwrap


def main():
    msg = textwrap.dedent("""
    DVSpeaker4Identification

    Main entry points:
      - python train.py --help
      - python eval.py --help
      - python eval_cross_condition.py --help
    """).strip()
    print(msg)


if __name__ == "__main__":
    main()
