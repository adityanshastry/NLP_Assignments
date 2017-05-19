from __future__ import division
import operator
import structperc


def main():
    tag_data = structperc.read_tagging_file("oct27.dev")

    tag_dict = {}
    for tags_list in tag_data:
        for tag in tags_list[1]:
            if tag not in tag_dict:
                tag_dict[tag] = 0
            tag_dict[tag] += 1

    print "Most common tag is ", max(tag_dict.iteritems(), key=operator.itemgetter(1))[0]
    print "Accuracy is ", tag_dict['V'] / sum(tag_dict.values()) * 100


if __name__ == '__main__':
    main()
