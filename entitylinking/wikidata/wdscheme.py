from entitylinking.utils import load_list, RESOURCES_FOLDER, load_property_labels

WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
WIKIPEDIA_PREFIX = "https://en.wikipedia.org/wiki/"
property_blacklist = load_list(RESOURCES_FOLDER + "property_blacklist.txt")
property2label = load_property_labels(RESOURCES_FOLDER + "properties_with_labels.txt")
BLACKSET_PROPERTY_OBJECT_TYPES = {'commonsmedia',
                                  'external-id',
                                  'globe-coordinate',
                                  'math',
                                  'monolingualtext',
                                  'quantity',
                                  'string',
                                  'url',
                                  'wikibase-property'}

content_properties = {p for p, v in property2label.items() if v.get("type") not in BLACKSET_PROPERTY_OBJECT_TYPES
                       and v.get('freq') > 5
                       and 'category' not in v.get('label', "")
                       and 'taxon' not in v.get('label', "")} - property_blacklist

frequent_properties = {p for p in content_properties if property2label[p].get('freq') > 1000}
