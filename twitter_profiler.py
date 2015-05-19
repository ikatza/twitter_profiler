# -*- coding: utf-8 -*-

import cv2
import numpy as np
# from matplotlib import pyplot as plt
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import urlparse
# import Image
import math
import requests
# from PIL import Image
# from StringIO import StringIO
import argparse
import json
import time
import threading
import logging
import unidecode
import unicodedata
import re

acceptance = {}
acceptance['ent_small'] = 0.87
acceptance['mean_small'] = 0.7
acceptance['all_small'] = 0.86
acceptance['ent_original'] = 0.77
acceptance['mean_original'] = 0.6
acceptance['all_original'] = 0.74
acceptance['extra_ent'] = 0.09
acceptance['extra_all'] = 0.08

params_dict = {}
params_dict['bins'] = 256
params_dict['colors'] = 'BGR'
params_dict['filter_size'] = 6
params_dict['iterations'] = 5


class ActivePool(object):
    def __init__(self):
        super(ActivePool, self).__init__()
        self.active = []
        self.lock = threading.Lock()

    def makeActive(self, name):
        with self.lock:
            self.active.append(name)
            # logging.debug('Running: %s', self.active)

    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)
            # logging.debug('Running: %s', self.active)


# def worker(path, imgs_folder, image_classifiers_data, summary, s1, s2, pool):
# def worker(path, imgs_folder, image_classifiers_data, summary, s2, pool):
def worker(screen_name, idx, image_classifiers_data, results, s, pool):
    # logging.debug('Waiting to join the pool')
    # s1.acquire()
    # logging.debug('Loading cascades')
    # cascades = load_cascades('haarcascades/')
    with s:
        name = threading.currentThread().getName()
        pool.makeActive(name)
        # time.sleep(0.1)
        # img_path = imgs_folder + '/' + path
        # if os.path.splitext(img_path)[1] == '':
        #     summary['undetermined'] += 1
        #     pool.makeInactive(name)
        img_type = loop_tester(screen_name,
                               image_classifiers_data,
                               # cascades,
                               verbose)
        # if img_type == 'logo':
        #     new_path = imgs_folder + '/' + 'logos' + '/' + path
        #     os.rename(img_path, new_path)
        #     summary['logos'] += 1
        # elif img_type == 'photo':
        #     new_path = imgs_folder + '/' + 'photos' + '/' + path
        #     os.rename(img_path, new_path)
        #     summary['photos'] += 1
        # else:
        #     # print "Could not determine image type."
        #     summary['undetermined'] += 1

        # If it's a photo then is a person.
        # If it's a logo then is a non_person.
        if img_type == 'undetermined':
            # print " "
            results[idx]['result'] = 'Null'
        elif img_type == 'photo':
            # print " "
            results[idx]['result'] = 'Person'
            results[idx]['method'] = 'Image type'
        elif img_type == 'logo':
            # print " "
            results[idx]['result'] = 'NotPerson'
            results[idx]['method'] = 'Image type'
        # s1.release()
        pool.makeInactive(name)


# def loop_tester(img_path, image_classifiers_data, cascades, verbose):
def loop_tester(screen_name, image_classifiers_data, verbose):
    # tests = ['ent', 'blur_diff_me', 'blur_diff_med', 'blur_diff_var']
    blur_diff_tests = ['blur_diff_me', 'blur_diff_med', 'blur_diff_var']
    sizes = ['small', 'original']
    # sizes = ['original']
    # sizes = ['small']
    # sum_logo_ratio = 0
    # sum_photo_ratio = 0
    ratios = []
    for size in sizes:
        probs = []
        # screen_name = os.path.splitext(os.path.basename(img_path))[0]
        if size == "original":
            image_url = 'https://twitter.com/' + \
                    screen_name + \
                    '/profile_image?size=original'
        elif size == "small":
            image_url = 'https://twitter.com/' + \
                    screen_name + \
                    '/profile_image?size=bigger'
        image = url_to_cv2image(image_url)
        # image = url_to_cv2image_long(image_url)
        if image is None:
            img_type = 'undetermined'
            return img_type

        # threadLock.acquire()
        face = detect_faces(image, cascades)
        # threadLock.release()
        if face > 0.:
            img_type = 'photo'
            logging.debug('++ face'+screen_name+str(face)+img_type)
            return img_type

        ent_value = image_entropy(image, params_dict, verbose)
        if ent_value is not None:
            p = compare_classifier(ent_value,
                                       'ent',
                                       size,
                                       image_classifiers_data)
            probs.append(p)
            # sum_logo_ratio += p['logo_ratio']
            # sum_photo_ratio += p['photo_ratio']
            # ratios_diff = abs(p['logo_ratio']-p['photo_ratio'])
            # if ratios_diff > acceptance['ent'+'_'+size]:
            ratio = (p['logo_ratio']-p['photo_ratio']) \
              / (p['logo_ratio']+p['photo_ratio'])
            ratios.append(ratio)
            if ratio > acceptance['ent'+'_'+size] + acceptance['extra_ent']:
                img_type = 'logo'
                if verbose:
                    logging.debug('++ ent '+size+str(ent_value)+screen_name+img_type+str(ratio))
                return img_type
            elif ratio < -1.*acceptance['ent'+'_'+size]:
                img_type = 'photo'
                if verbose:
                    logging.debug('++ ent '+size+str(ent_value)+screen_name+img_type+str(ratio))
                return img_type
        if verbose:
            logging.debug('-- ent not concl '+size+str(ent_value)+screen_name+str(ratio))

        # Run blur_diff if no img_type has been found
        blur_diff = blur_difference(image, params_dict, verbose)
        if blur_diff is not None:
            for blur_diff_test in blur_diff_tests:
                if blur_diff_test == 'blur_diff_me':
                    statname = 'mean'
                elif blur_diff_test == 'blur_diff_med':
                    statname = 'median'
                elif blur_diff_test == 'blur_diff_var':
                    statname = 'variance'
                p = compare_classifier(blur_diff[statname],
                                       blur_diff_test,
                                       size,
                                       image_classifiers_data)
                probs.append(p)
                # sum_logo_ratio += p['logo_ratio']
                # sum_photo_ratio += p['photo_ratio']
                ratio = (p['logo_ratio']-p['photo_ratio']) \
                  / (p['logo_ratio']+p['photo_ratio'])
                ratios.append(ratio)
            ratios_mean = np.mean(ratios)
            ratios_max = max(ratios, key=abs)
            if ratios_mean > acceptance['mean'+'_'+size]:
                img_type = 'logo'
                if verbose:
                    logging.debug('++ mean '+size+screen_name+str(ratios_max)+img_type+str(ratios_mean))
                return img_type
            elif ratios_mean < -1.*acceptance['all'+'_'+size]:
                img_type = 'photo'
                if verbose:
                    logging.debug('++ mean '+size+screen_name+str(ratios_max)+img_type+str(ratios_mean))
                return img_type
            if ratios_max > acceptance['all'+'_'+size] + acceptance['extra_all']:
                img_type = 'logo'
                if verbose:
                    logging.debug('++ max '+size+screen_name+str(ratios_max)+img_type+str(ratios_mean))
                return img_type
            elif ratios_max < -1.*acceptance['all'+'_'+size]:
                img_type = 'photo'
                if verbose:
                    logging.debug('++ max '+size+screen_name+str(ratios_max)+img_type+str(ratios_mean))
                return img_type
        if verbose:
            logging.debug('-- Tests were not conclusive with '+str(blur_diff)+size+str(ent_value)+screen_name+str(probs)+str(ratios_mean)+' '+str(ratios_max)+'\n')
    img_type = 'undetermined'
    return img_type


def url_to_cv2image(url):
    try:
        r = requests.get(url)
        image = np.asarray(bytearray(r.content), dtype="uint8")
        # image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        return None
    return image


def detect_faces(img, cascades, verbose=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for cascade in cascades:
        idx = cascades.index(cascade)
        threadLocks[idx].acquire()
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        threadLocks[idx].release()
        if len(faces) > 0:
            break
    if len(faces) == 0:
        return False
    if len(faces) > 0:
        return True


def image_entropy(img, params_dict, verbose=False):
    try:
        if params_dict['colors'] == 'HSV':
            channels = img.shape[2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif params_dict['colors'] == 'BGR':
            channels = img.shape[2]
        elif params_dict['colors'] == 'Gray':
            channels = 1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except IndexError:
        channels = 1
    histogram = []
    for i in range(channels):
        histogram.extend(cv2.calcHist([img], [i], None,
                                      [params_dict['bins']],
                                      [0, 256]))
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    entropy = -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
    return float(entropy[0])


def blur_difference(img, params_dict, verbose=False):
    # if img is None:
    #     if '.gif' == os.path.splitext(path)[1]:
    #         pil_image = Image.open(path).convert('RGB')
    #         cv_image = np.array(pil_image)
    #         img = cv_image[:, :, ::-1].copy()
    #     else:
    #         print "Can't open ", path, " image, sorry."
    #         # return 0.
    #         return None
    rad = 150
    min_dim = min(img.shape[0], img.shape[1])
    if min_dim < rad:
        rad = min_dim
    filter_size = params_dict['filter_size']
    iterations = params_dict['iterations']
    blurred = cv2.bilateralFilter(img, filter_size, rad, rad)
    for i in range(0, iterations):
        blurred = cv2.bilateralFilter(blurred, filter_size,
                                      rad, rad)
    stats = {}
    stats['median'] = np.median(img - blurred)
    stats['mean'] = np.mean(img - blurred)
    stats['variance'] = np.var(img - blurred)
    # if verbose:
    #     # print path
    #     print "Median =", stats['median'], "\t",
    #     print "Mean =", stats['mean'], "\t",
    #     # print "Std Deviation = ", stats['std'], "\t",
    #     print "Variance = ", stats['variance']
    #     img_flip = img[:, :, ::-1]
    #     blur_flip = blurred[:, :, ::-1]
    #     plt.subplot(121), plt.imshow(img_flip), plt.title('Original')
    #     plt.xticks([]), plt.yticks([])
    #     plt.subplot(122), plt.imshow(blur_flip), plt.title('Bilateral')
    #     plt.xticks([]), plt.yticks([])
    #     plt.show()
    return stats


def detected_gender_init(name, screen_name):
    detected_gender = {}
    detected_gender['name'] = name
    detected_gender['screen_name'] = screen_name
    detected_gender['male'] = 0
    detected_gender['female'] = 0
    detected_gender['person'] = 0
    detected_gender['not_person'] = 0
    detected_gender['first_gender'] = None
    detected_gender['method'] = None
    detected_gender['result'] = None
    return detected_gender


def load_image_classifier_data(folder):
    image_classifiers_data = {}
    for path in os.listdir(folder):
        filename = path[:-5]
        path = folder + '/' + path
        classifier_file = open(path)
        classifier_json = classifier_file.read()
        classifier_file.close()
        classifier_list = json.loads(classifier_json)
        image_classifiers_data[filename] = classifier_list
    return image_classifiers_data


def load_cascades(folder):
    c = []
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_frontalface_default.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_frontalface_alt.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_frontalface_alt2.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_frontalface_alt_tree.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_mcs_mouth.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_profileface.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_eye.xml'))
    c.append(cv2.CascadeClassifier(
        folder+'haarcascade_eye_tree_eyeglasses.xml'))
    return c


def compare_classifier(value, statistic, image_size, image_classifiers_data):
    if(statistic != 'blur_diff_me' and
       statistic != 'blur_diff_med' and
       statistic != 'blur_diff_var' and
       statistic != 'ent'):
        print "No proper statistic, options are: ",
        print 'blur_diff_me,', 'blur_diff_med,', 'blur_diff_var,', 'ent'
        return
    if(image_size != 'original' and
       image_size != 'small'):
        print "No proper image_size, options are: ", 'original,', 'small'
    dist_name = statistic + '_' + image_size
    len_photos = len(image_classifiers_data[dist_name]['photos'])
    len_logos = len(image_classifiers_data[dist_name]['logos'])
    probs = {}
    if value >= image_classifiers_data[dist_name]['photos'][-1]:
        probs['photo_ratio'] = 1.
    else:
        for idx, val in enumerate(
                image_classifiers_data[dist_name]['photos'],
                start=1):
            if value < val:
                probs['photo_ratio'] = float(idx)/float(len_photos)
                break
    if value > image_classifiers_data[dist_name]['logos'][-1]:
        probs['logo_ratio'] = 1./float(len_logos)
    elif value <= image_classifiers_data[dist_name]['logos'][0]:
        probs['logo_ratio'] = 1.
    else:
        for idx, val in reversed(
                list(enumerate(image_classifiers_data[dist_name]['logos']))):
            if value > val:
                probs['logo_ratio'] = float(len_logos-idx)/float(len_logos)
                break
    return probs


def test_genders(name, detected_gender):
    name_split = camel_to_blanks(name).split()
    for s in name_split:
        if len(s) > 2:
            # print s, gender
            ss = remove_accents(s)
            gender = Genderizer.detect(firstName=ss, lang='es')
            if not gender:  # Expand for other categories
                unknown = 1
            elif gender == 'male':
                detected_gender['male'] += 1
                if not detected_gender['first_gender']:
                    detected_gender['first_gender'] = 'male'
            elif gender == 'female':
                detected_gender['female'] += 1
                if not detected_gender['first_gender']:
                    detected_gender['first_gender'] = 'female'
            elif gender == 'person':
                detected_gender['person'] += 1
    return detected_gender


def remove_accents(input_str):
    nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def slugify(input_str):
    string = unidecode.unidecode(unicode(input_str))
    return string


def camel_to_blanks(camelCasedStr):
    """
    This function converts to blank spaces from camelCase
    """
    _first_cap_re = re.compile(r'(.)([A-Z][a-z]+)')
    _all_cap_re = re.compile('([a-z0-9])([A-Z])')
    sub1 = _first_cap_re.sub(r'\1 \2', camelCasedStr)
    blank_spaced_str = _all_cap_re.sub(r'\1 \2', sub1).lower()
    blank_spaced_str = blank_spaced_str.replace('__', ' ')
    blank_spaced_str = blank_spaced_str.replace('_', ' ')
    blank_spaced_str = blank_spaced_str.replace('.', ' ')
    blank_spaced_str = blank_spaced_str.replace('-', ' ')
    blank_spaced_str = blank_spaced_str.replace("'", ' ')
    return blank_spaced_str


def result(detected_gender):
    # if detected_gender['male']-detected_gender['female'] == 0 \
    #   and detected_gender['first_gender'] is None:
    if detected_gender['first_gender'] is None:
        detected_gender['method'] = None
        detected_gender['result'] = 'Null'
        # print "Null", detected_gender['name'], "\t\t", \
        #   detected_gender['screen_name']
    if(detected_gender['male']-detected_gender['female'] == 0
       and detected_gender['first_gender'] == 'male'):
        detected_gender['method'] = '1st gender'
        detected_gender['result'] = 'Male'
        # print "Male 2R", detected_gender['name'], "\t\t", \
        #   detected_gender['screen_name']
    if(detected_gender['male']-detected_gender['female'] == 0
       and detected_gender['first_gender'] == 'female'):
        detected_gender['method'] = '1st gender'
        detected_gender['result'] = 'Female'
        # print "Female 2R", detected_gender['name'], "\t\t", \
        #   detected_gender['screen_name']
    if detected_gender['male']-detected_gender['female'] > 0:
        detected_gender['method'] = 'Majority of names'
        detected_gender['result'] = 'Male'
        # print "Male", detected_gender['name'], "\t\t", \
        #   detected_gender['screen_name']
    if detected_gender['male'] - detected_gender['female'] < 0:
        detected_gender['method'] = 'Majority of names'
        detected_gender['result'] = 'Female'
        # print "Female", detected_gender['name'], "\t\t", \
        #   detected_gender['screen_name']
    if(detected_gender['male'] == 0 and
       detected_gender['female'] == 0 and
       detected_gender['person'] > 0):
        detected_gender['method'] = 'Person identifier'
        detected_gender['result'] = 'Person'
        print "Person", detected_gender['name'], "\t\t",
        print detected_gender['screen_name']
    return detected_gender


def url_extension(url):
    path = urlparse.urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext


def backwards_search(name, detected_gender):
    normalized_name = slugify(name).lower()
    # Empirical best numbers for length of names
    longest = 15  # No way it could be longer
    at_least = 5  # If shorter introduces many errors
    if len(normalized_name) < longest:
        longest = len(normalized_name)
    chars = list(reversed(range(at_least, longest)))
    sub_strings = []
    for char in chars:
        sub_strings.append(normalized_name[0:char])
    for sub_str in sub_strings:
        gender = Genderizer.detect(firstName=sub_str,
                                   lang='es')
        if gender == 'male':
            detected_gender['method'] = 'Sub-string'
            detected_gender['result'] = 'Male'
            # print gender, usr['screen_name'], sub_str
            break
        if gender == 'female':
            detected_gender['method'] = 'Sub-string'
            detected_gender['result'] = 'Female'
            # print gender, usr['screen_name'], sub_str
            break
        if gender == 'person':
            detected_gender['method'] = 'Sub-string'
            detected_gender['result'] = 'Person'
            print gender, usr['screen_name'], sub_str
            break
    return detected_gender


def print_summary_results(result):
    null = 0
    notperson = 0
    notperson_logo = 0
    person = 0
    person_1R = 0
    person_2R = 0
    person_3R = 0
    male = 0
    female = 0
    male_1R = 0
    male_2R = 0
    male_3R = 0
    female_1R = 0
    female_2R = 0
    female_3R = 0
    for result in results:
        if result['result'] == 'Null':
            null += 1
            print "Null", result['name'], "\t\t", result['screen_name']
        elif result['result'] == 'NotPerson':
            notperson += 1
            if result['method'] == 'Image type':
                notperson_logo += 1
        elif result['result'] == 'Person':
            person += 1
            if result['method'] == 'Person identifier':
                person_1R += 1
            elif result['method'] == 'Sub-string':
                person_2R += 1
            elif result['method'] == 'Image type':
                person_3R += 1
        elif result['result'] == 'Male':
            male += 1
            if result['method'] == 'Majority of names':
                male_1R += 1
            elif result['method'] == '1st gender':
                male_2R += 1
            elif result['method'] == 'Sub-string':
                male_3R += 1
        elif result['result'] == 'Female':
            female += 1
            if result['method'] == 'Majority of names':
                female_1R += 1
            elif result['method'] == '1st gender':
                female_2R += 1
            elif result['method'] == 'Sub-string':
                female_3R += 1
    print "\n"
    print "len(results)", len(results)
    print "Is it the same than the input? ",
    print len(results) == len(tw_names_list)

    print "Null \t", null
    print "NotPerson \t", notperson

    print "Persons \t", person
    print "Persons 1R \t", person_1R, "\t",
    print "Persons 2R \t", person_2R
    print "Persons adds correctly: \t",
    print person == person_1R+person_2R

    print "Males \t", male
    print "Males 1R \t", male_1R, "\t",
    print "Males 2R \t", male_2R, "\t",
    print "Males 3R \t", male_3R
    print "Males adds correctly: \t",
    print male == male_1R+male_2R+male_3R

    print "Females \t", female
    print "Females 1R \t", female_1R, "\t",
    print "Females 2R \t", female_2R, "\t",
    print "Females 3R \t", female_3R
    print "Females adds correctly: \t",
    print female == female_1R+female_2R+female_3R

    print "Everything adds correctly:  \t",
    print len(tw_names_list) == male + female + null + notperson + person
    print "------------------------------ \n\n"



class NamesCollection(object):
    """
    This is a primitive interface for the collection of first names.

    When the first query is received, it reads all the first
    names and their extra information from source file. So this step
    may take some time.

    After the first request and successful importing, it serves the
    data from memory.

    For more than one request, this class would be ok. But you have
    better choose a reasonable interface such as memcache interface or
    mongodb interface.
    """

    isInitialized = None

    # collectionSourceFile = 'data/name_gender.csv'
    collectionSourceFile = 'data/name_gender_mex.csv'
    collection = None

    @classmethod
    def init(cls):
        if not cls.collection:
            cls.collection = cls.loadFromSource()

    @classmethod
    def loadFromSource(cls):
        # will produce something like this:
        # {'kamil': {'tr': 'm'}, 'mustafa': {'en': 'm', 'tr': 'm'}, 'kim': {'sk': 'm', 'nl': 'f'}}
        items = {}
        path = os.path.abspath(__file__)
        # dir_path = os.path.dirname(path) + '/'
        dir_path = os.path.dirname(path)
        for i in open(dir_path + '/' + cls.collectionSourceFile):
            item = i.strip().split(',')
            firstName = item[0].lower()
            if len(item) == 2:
                # item.append('en')
                item.append('es')
            # print item
            item = {item[2]: item[1]}
            if firstName in items:
                items[firstName] = dict(items[firstName].items()+item.items())
            else:
                items[firstName] = item
        return items
    @classmethod
    # def getGender(cls, firstName, lang='en'):
    def getGender(cls, firstName, lang='es'):
        if not cls.isInitialized:
            cls.init()
        firstName = firstName.lower()
        nameInfo = cls.collection.get(firstName, None)
        if not nameInfo:
            return None

        if nameInfo.get(lang, None):
            return {'name': firstName, 'gender': nameInfo[lang], 'lang': lang}
        # elif nameInfo.get('en', None):
            # return {'name': firstName, 'gender': nameInfo['en'], 'lang': 'en'}
        elif nameInfo.get('es', None):
            return {'name': firstName, 'gender': nameInfo['es'], 'lang': 'es'}
        else:
            return None


namesCollection = NamesCollection()


class Genderizer(object):
    """ @TODO: write docstring for Genderize"""
    initialized = False
    namesCollection = None
    # classifier = None
    # lang = 'tr'
    lang = 'es'
    significantDegree = 0.3
    surelyMale = 'M'
    surelyFemale = 'F'
    surelyPerson = 'P'
    mostlyMale = '?m'
    mostlyFemale = '?f'
    genderUnknown = '?'

    @classmethod
    def init(cls,
             # lang = 'tr',
             lang='es',
             namesCollection=NamesCollection,
             # classifier=None
             ):
        cls.lang = lang
        cls.namesCollection = namesCollection
        # if classifier:
        #     cls.classifier = classifier
        # else:
        #     cls.classifier = Classifier(CachedModel.get(lang), tokenizer)
        cls.initialized = True

    @classmethod
    def detect(cls, firstName=None, text=None, lang=None):
        if not cls.initialized:
            cls.init()
        # if cls.classifier is None:
        #     raise Exception('No classifier found. You need to set one.')
        if cls.namesCollection is None:
            raise Exception('No names collection found. You need to have one.')
        if firstName:
            nameGender = cls.namesCollection.getGender(firstName, lang)
            # if the first name surely is used for only one gender,
            # we can accept this gender.
            if nameGender:
                if nameGender['gender'] == cls.surelyMale:
                    return 'male'
                elif nameGender['gender'] == cls.surelyFemale:
                    return 'female'
                elif nameGender['gender'] == cls.surelyPerson:
                    return 'person'
        else:
            nameGender = None
        # It is not for sure which gender the first name is being used in
        # we try to detect it looking his/her writing style.
        if text:
            probablities = dict(cls.classifier.classify(text))
            # @TODO: NEJDET, you have better explain what we are doing here,
            # why making log calculation and what is these ratios.
            #classifierScoreLogF = math.log(probablities['female']) / math.log(sum(probablities.values()))
            #classifierScoreLogM = math.log(probablities['male']) / math.log(sum(probablities.values()))
            classifierScoreLogF = probablities['female'] / sum(probablities.values())
            classifierScoreLogM = probablities['male'] / sum(probablities.values())
            classifierScoreM = classifierScoreLogF / (classifierScoreLogM + classifierScoreLogF)
            classifierScoreF = classifierScoreLogM / (classifierScoreLogM + classifierScoreLogF)
            if nameGender and nameGender['gender'].startswith('?'):
                if nameGender['gender'] == cls.mostlyMale \
                  and classifierScoreM > 0.6:
                    return 'male'
                elif nameGender['gender'] == cls.mostlyFemale \
                  and classifierScoreF > 0.6:
                    return 'female'
                elif nameGender['gender'] != cls.genderUnknown:
                    return None
            # If there is no information according to the name and
            # there is significant difference between the two probablity,
            # we can accept the highest probablity.
            if abs(classifierScoreF - classifierScoreM) > cls.significantDegree:
                if probablities['female'] > probablities['male']:
                    return 'female'
                else:
                    return 'male'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify images in folder and move them to'
        'photo-like or logo-like sub-folders.')
    # parser.add_argument('-f', '--folder',
    #                     help='Input images folder path',
    #                     required=True)
    # parser.add_argument('-t', '--type',
    #                     help='Type of images',
    #                     required=False)
    parser.add_argument('-s', '--screen_name',
                        help='User screen name',
                        # action='store_true',
                        # default=False,
                        required=True)
    parser.add_argument('-v', '--verbose',
                        help='Be verbose',
                        action='store_true',
                        # default=False,
                        required=False)
    args = parser.parse_args()
    verbose = args.verbose
    screen_name = args.screen_name

    image_classifiers_data = load_image_classifier_data(
        'classifiers_empirical_dists')
    cascades = load_cascades('haarcascades/')

    pool = ActivePool()
    active_threads = 20
    # s1 = threading.Semaphore(active_threads+10)
    s = threading.Semaphore(active_threads)
    threadLocks = []
    for idx in range(len(cascades)):
        threadLocks.append(threading.Lock())
    threads = []

    file_names = ['tw_names_follws_'+screen_name+'.json',
                  'tw_names_friends_'+screen_name+'.json']
    for f in file_names:
        tw_names_file = open(f)
        tw_names_json = tw_names_file.read()
        tw_names_file.close()
        tw_names_list = json.loads(tw_names_json)
        print screen_name
        print tw_names_file, len(tw_names_list)

        results = []
        for usr in tw_names_list:
            detected_gender = detected_gender_init(usr['name'],
                                                   usr['screen_name'])
            detected_gender = test_genders(usr['name'],
                                           detected_gender)
            detected_gender = test_genders(usr['screen_name'],
                                           detected_gender)
            detected_gender = result(detected_gender)
            if detected_gender['result'] == 'Null':
                detected_gender = backwards_search(usr['screen_name'],
                                                   detected_gender)
            if detected_gender['result'] == 'Null':
                detected_gender = backwards_search(usr['name'],
                                                   detected_gender)
            results.append(detected_gender.copy())

        # Try to determine gender by looking at the profile image.
        for r in results:
            if r['result'] != 'Null':
                continue
            idx = results.index(r)
            t = threading.Thread(target=worker,
                                 name=r['screen_name'],
                                 args=(r['screen_name'],
                                       idx,
                                       image_classifiers_data,
                                       results, s, pool))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        print_summary_results(results)
