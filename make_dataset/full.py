# README
# Phillip Long
# July 17, 2024

# Parse through all musescore (.mscz) files and determine which are in the public domain.

# python /home/pnlong/model_musescore/make_dataset/full.py


# IMPORTS
##################################################

import glob
from os.path import isfile, exists, basename, dirname, realpath
from os import makedirs
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse
import logging
from re import sub
import json
import math
import muspy

import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from read_mscz.read_mscz import read_musescore, get_musescore_version
from read_mscz.music import MusicExpress
import model_remi.representation
import utils

##################################################


# CONSTANTS
##################################################

MUSESCORE_DIR = "/data2/zachary/musescore"
INPUT_DIR = "/data2/pnlong/musescore"
METADATA_MAPPING = f"{INPUT_DIR}/metadata_to_data.csv"
DATASET_DIR_NAME = "dataset"
OUTPUT_DIR = f"{INPUT_DIR}/{DATASET_DIR_NAME}"
LIST_FEATURE_JOIN_STRING = "-"
MMT_STATISTIC_COLUMNS = ["pitch_class_entropy", "scale_consistency", "groove_consistency"] # names of the MMT-style statistics

# public domain licenses for extracting from metadata
PUBLIC_LICENSE_URLS = (
    "https://creativecommons.org/publicdomain/mark/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    )
DEFAULT_LICENSE = "publicdomain"
DEFAULT_LICENSE_URL = PUBLIC_LICENSE_URLS[0]

# for multiprocessing
CHUNK_SIZE = 1

##################################################


# MMT-STYLE SONG STATISTICS
##################################################

def pitch_class_entropy(music: MusicExpress) -> float:
    """Return the entropy of the normalized note pitch class histogram.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#pitch_class_entropy

    The pitch class entropy is defined as the Shannon entropy of the
    normalized note pitch class histogram. Drum tracks are ignored.
    Return NaN if no note is found. This metric is used in [1].

    .. math::
        pitch\_class\_entropy = -\sum_{i = 0}^{11}{
            P(pitch\_class=i) \times \log_2 P(pitch\_class=i)}

    Parameters
    ----------
    music : :class:`read_mscz.MusicExpress`
        Music object to evaluate.

    Returns
    -------
    float
        Pitch class entropy.

    See Also
    --------
    :func:`muspy.pitch_entropy` :
        Compute the entropy of the normalized pitch histogram.

    References
    ----------
    1. Shih-Lun Wu and Yi-Hsuan Yang, "The Jazz Transformer on the Front
       Line: Exploring the Shortcomings of AI-composed Music through
       Quantitative Measures”, in Proceedings of the 21st International
       Society for Music Information Retrieval Conference, 2020.

    """
    counter = np.zeros(12)
    for track in music.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            counter[note.pitch % 12] += 1
    denominator = counter.sum()
    if denominator < 1:
        return math.nan
    prob = counter / denominator
    return muspy.metrics.metrics._entropy(prob = prob)

def scale_consistency(music: MusicExpress) -> float:
    """Return the largest pitch-in-scale rate.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#scale_consistency

    The scale consistency is defined as the largest pitch-in-scale rate
    over all major and minor scales. Drum tracks are ignored. Return NaN
    if no note is found. This metric is used in [1].

    .. math::
        scale\_consistency = \max_{root, mode}{
            pitch\_in\_scale\_rate(root, mode)}

    Parameters
    ----------
    music : :class:`read_mscz.MusicExpress`
        Music object to evaluate.

    Returns
    -------
    float
        Scale consistency.

    See Also
    --------
    :func:`muspy.pitch_in_scale_rate` :
        Compute the ratio of pitches in a certain musical scale.

    References
    ----------
    1. Olof Mogren, "C-RNN-GAN: Continuous recurrent neural networks
       with adversarial training," in NeuIPS Workshop on Constructive
       Machine Learning, 2016.

    """
    max_in_scale_rate = 0.0
    for mode in ("major", "minor"):
        for root in range(12):
            rate = muspy.metrics.metrics.pitch_in_scale_rate(music = music, root = root, mode = mode)
            if math.isnan(rate):
                return math.nan
            if rate > max_in_scale_rate:
                max_in_scale_rate = rate
    return max_in_scale_rate

def groove_consistency(music: MusicExpress) -> float:
    """Return the groove consistency.
    Copied from https://salu133445.github.io/muspy/_modules/muspy/metrics/metrics.html#groove_consistency

    The groove consistency is defined as the mean hamming distance of
    the neighboring measures.

    .. math::
        groove\_consistency = 1 - \frac{1}{T - 1} \sum_{i = 1}^{T - 1}{
            d(G_i, G_{i + 1})}

    Here, :math:`T` is the number of measures, :math:`G_i` is the binary
    onset vector of the :math:`i`-th measure (a one at position that has
    an onset, otherwise a zero), and :math:`d(G, G')` is the hamming
    distance between two vectors :math:`G` and :math:`G'`. Note that
    this metric only works for songs with a constant time signature.
    Return NaN if the number of measures is less than two. This metric
    is used in [1].

    Parameters
    ----------
    music : :class:`read_mscz.MusicExpress`
        Music object to evaluate.
    measure_resolution : int
        Time steps per measure.

    Returns
    -------
    float
        Groove consistency.

    References
    ----------
    1. Shih-Lun Wu and Yi-Hsuan Yang, "The Jazz Transformer on the Front
       Line: Exploring the Shortcomings of AI-composed Music through
       Quantitative Measures”, in Proceedings of the 21st International
       Society for Music Information Retrieval Conference, 2020.

    """

    measure_resolution = 4 * music.resolution
    length = max([note.time + note.duration for track in music.tracks for note in track.notes] + [0])
    if measure_resolution < 1:
        raise ValueError("Measure resolution must be a positive integer.")

    n_measures = int(length / measure_resolution) + 1
    if n_measures < 2:
        return math.nan

    groove_patterns = np.zeros(shape = (n_measures, measure_resolution), dtype = bool)

    for track in music.tracks:
        for note in track.notes:
            measure, position = map(int, divmod(note.time, measure_resolution)) # ensure these values are integers, as they will be used for indexing
            if not groove_patterns[measure, position]:
                groove_patterns[measure, position] = 1

    hamming_distance = np.count_nonzero(a = (groove_patterns[:-1] != groove_patterns[1:]))

    return 1 - (hamming_distance / (measure_resolution * (n_measures - 1)))

##################################################


# HELPER FUNCTIONS
##################################################

# help parse a list feature from a metadata file
extract_letters = lambda item: sub(pattern = "[^a-z]", repl = "", string = item.lower()) # extract letters from a string, convert to lower case
is_string_occupied = lambda item: (len(item) > 0) and (item != "none") # is a string not empty
def get_list_feature_string(list_feature: list) -> str:
    """
    Convert a list feature from metadata into a single string.
    """
    list_feature = map(extract_letters, list_feature) # extract letters, convert to lower case
    list_feature = filter(is_string_occupied, list_feature) # remove empty strings
    list_feature_string = LIST_FEATURE_JOIN_STRING.join(list_feature).strip() # store list feature as a single string
    if len(list_feature_string) == 0:
        list_feature_string = None
    return list_feature_string

# get tracks string
def get_tracks_string(tracks: list) -> str:
    """
    Convert a Music object's tracks into a single string.
    """
    return LIST_FEATURE_JOIN_STRING.join(map(str, sorted(map(lambda track: track.program, tracks))))

##################################################


# FUNCTION FOR PARSING THROUGH EACH FILE
##################################################

def get_full_dataset(path: str) -> None:
    """
    Given a MuseScore filepath, determine if that file is in our dataset (public domain),
    and then further analyze said file with more metrics.

    Parameters
    ----------
    path : str
        MuseScore filepath to operate on

    Returns
    -------
    void

    """

    # DETERMINE IF SONG IS VALID FOR OUR DATASET
    ##################################################

    # determine public domain
    is_public_domain = False # assume a file is not in the public domain to ere on the conservative side
    metadata_path = METADATA.get(path, None) # try to get the metadata path
    if metadata_path:
        try: # use a try in case there are errors opening the metadata file
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(fp = metadata_file)
            license_url = metadata["data"].get("license_url", "")
            is_public_domain = (license_url in PUBLIC_LICENSE_URLS)
        except (OSError):
            metadata_path = None

    # determine if file is valid
    # if a file is copyrighted, it is automatically invalid, whether or not it opens
    # we only investigate public domain files to save time
    is_valid = False # default to invalid
    if is_public_domain: # if a song is in the public domain, we will investigate
        try: # try to read musescore
            music = read_musescore(path = path, timeout = 10)
            n_notes = sum(len(track.notes) for track in music.tracks)
            is_valid = (n_notes > 0) # check for empty songs
        except:
            pass

    # output results of whether a song is in the public domain to all files
    results_all = {
        "data": path,
        "metadata": metadata_path,
        "is_public_domain": is_public_domain,
        "is_valid": is_valid,
    }
    utils.write_to_file(info = results_all, columns = list(results_all.keys()), output_filepath = OUTPUT_FILEPATH_ALL)

    # stop execution if a track is invalid, meaning it is either copyrighted or it is public domain, but doesn't open correctly
    if not is_valid:
        return

    ##################################################


    # FURTHER ANALYZE VALID TRACKS, FIRST GETTING SOME BASIC INFO
    ##################################################
    # at this point, we have passed the valid gate
    # songs analyzed here are in the full, uncleaned dataset
    # in other words, songs are in the public domain and can open properly

    # start results dictionary
    results = {
        "path" :            path,
        "metadata" :        metadata_path,
        "has_metadata" :    bool(metadata_path),
    }

    # try to get the version
    try:
        results["version"] = get_musescore_version(path = path)
    except:
        results["version"] = None

    ##################################################


    # INVESTIGATE METADATA
    ##################################################
    # dive into metadata, see bottom of file for example of metadata file

    # set defaults
    results.update({
        "is_user_pro" :         False, # ere on the conservative side, assume user is not pro by default
        "is_user_publisher" :   False, # assume user is not a publisher by default
        "is_user_staff" :       False, # assume user is not part of MuseScore staff by default
        "has_paywall" :         False, # assume no paywall by default
        "is_rated" :            False, # assume no ratings
        "is_official" :         False, # assume song is not the official score by default
        "is_original" :         False, # assume song is not original by default
        "is_draft" :            False, # assume song is not a draft by default
        "has_custom_audio" :    False, # assume no custom audio by default
        "has_custom_video" :    False, # assume no custom video by default
        "n_comments" :          0, # assume no comments by default
        "n_favorites" :         0, # assume no one has favorited this song by default
        "n_views" :             0, # assume no views by default
        "n_ratings" :           0, # assume no one has rated the song by default
        "rating" :              0, # the default rating is 0, as the lowest possible rating is 1
        "license" :             DEFAULT_LICENSE, # assume public domain by default
        "license_url" :         DEFAULT_LICENSE_URL, # assume public domain by default
        "genres" :              None, # assume no genre labels by default
        "groups" :              None, # assume no group labels by default
        "tags" :                None, # assume no tag labels by default
        "song_name" :           None, # assume no song name by default
        "title" :               None, # assume no title by default
        "subtitle" :            None, # assume no subtitle by default
        "artist_name" :         None, # assume unknown artist by default
        "composer_name" :       None, # assume unknown composer by default
        "publisher" :           None, # assume song has no publisher by default
        "complexity" :          None, # assume no complexity by default
    })

    # if there exists an associated metadata path
    if metadata_path:

        # open metadata
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(fp = metadata_file)

        # get metadata features
        results["complexity"] = int(metadata["data"].get("complexity", None))
        results["n_comments"] = int(metadata["data"].get("count_comments", 0))
        results["n_favorites"] = int(metadata["data"].get("count_favorites", 0))
        results["n_views"] = int(metadata["data"].get("count_views", 0))
        results["genres"] = get_list_feature_string(list_feature = list(map(lambda genre: str(genre.get("name", "")), metadata["data"].get("genres", []))))
        results["groups"] = get_list_feature_string(list_feature = list(map(lambda group: str(group.get("title", "")), metadata["data"].get("groups", []))))
        results["license_url"] = metadata["data"].get("license_url", DEFAULT_LICENSE_URL)
        if ("paywall" in metadata["data"].keys()) and (metadata["data"]["paywall"] is not None):
            results["has_paywall"] = bool(metadata["data"]["paywall"].get("has_instant_paywall", False))
        if ("score" in metadata["data"].keys()) and (metadata["data"]["score"] is not None):
            results["artist_name"] = metadata["data"]["score"].get("artist_name", None)
            results["composer_name"] = metadata["data"]["score"].get("composer_name", None)
            results["has_custom_audio"] = bool(metadata["data"]["score"].get("has_custom_audio", False))
            results["has_custom_video"] = bool(metadata["data"]["score"].get("has_custom_video", False))
            results["is_draft"] = bool(metadata["data"]["score"].get("is_draft", False))
            results["is_official"] = bool(metadata["data"]["score"].get("is_official", False))
            results["is_original"] = bool(metadata["data"]["score"].get("is_original", False))
            results["license"] = metadata["data"]["score"].get("license", DEFAULT_LICENSE)
            results["publisher"] = metadata["data"]["score"].get("publisher", None)
            if ("rating" in metadata["data"]["score"].keys()) and (metadata["data"]["score"]["rating"] is not None):
                results["n_ratings"] = int(metadata["data"]["score"]["rating"].get("count", 0))
                results["rating"] = float(metadata["data"]["score"]["rating"].get("rating", 0))
                results["is_rated"] = bool(results["n_ratings"])
            results["song_name"] = metadata["data"]["score"].get("song_name", None)
            results["subtitle"] = metadata["data"]["score"].get("subtitle", None)
            results["tags"] = get_list_feature_string(list_feature = list(map(str, metadata["data"]["score"].get("tags", []))))
            results["title"] = metadata["data"]["score"].get("title", None)
            if ("user" in metadata["data"]["score"].keys()) and (metadata["data"]["score"]["user"] is not None):
                results["is_user_pro"] = bool(metadata["data"]["score"]["user"].get("is_pro", False))
                results["is_user_publisher"] = bool(metadata["data"]["score"]["user"].get("is_publisher", False))
                results["is_user_staff"] = bool(metadata["data"]["score"]["user"].get("is_staff", False))

    ##################################################


    # INVESTIGATE THE SONG ITSELF
    ##################################################
    # dive into song itself

    # obtain some helper variables
    n_annotations = sum(len(track.annotations) for track in music.tracks) + len(music.annotations) + len(music.tempos) + max(len(music.key_signatures) - 1, 0) + max(len(music.time_signatures) - 1, 0) + sum(barline.subtype != "single" for barline in music.barlines)
    n_lyrics = sum(len(track.lyrics) for track in music.tracks) + len(music.lyrics)
    
    # add findings to results
    results.update({
        "n_tracks" :                len(music.tracks),
        "tracks" :                  get_tracks_string(tracks = music.tracks),
        "song_length" :             music.song_length,
        "song_length.seconds" :     music.metrical_time_to_absolute_time(time_steps = music.song_length),
        "song_length.bars" :        len(music.barlines),
        "song_length.beats" :       len(music.beats),
        "n_notes" :                 n_notes,
        "notes_per_bar" :           n_notes / len(music.barlines),
        "n_annotations" :           n_annotations,
        "has_annotations" :         bool(n_annotations),
        "n_lyrics" :                n_lyrics,
        "has_lyrics" :              bool(n_lyrics),
        "n_tokens" :                n_notes + n_annotations + n_lyrics,
    })

    ##################################################


    # OUTPUT FINDINGS
    ##################################################

    # encode then decode
    notes = model_remi.representation.extract_notes(music = music, resolution = encoding["resolution"])
    notes = notes[notes[:, 0] < encoding["max_beat"]] # filter so that all beats are in the vocabulary
    notes[:, 2] = np.clip(a = notes[:, 2], a_min = 0, a_max = 127) # remove unknown pitches
    codes = model_remi.representation.encode_notes(notes = notes, encoding = encoding, indexer = indexer)
    notes = model_remi.representation.decode_notes(data = codes, encoding = encoding, vocabulary = vocabulary)
    music = model_remi.representation.reconstruct(notes = notes, resolution = encoding["resolution"])

    # extract MMT-style statistics
    results.update(dict(zip(MMT_STATISTIC_COLUMNS, (
        pitch_class_entropy(music = music),
        scale_consistency(music = music),
        groove_consistency(music = music),
    ))))

    # output results
    utils.write_to_file(info = results, columns = list(results.keys()), output_filepath = OUTPUT_FILEPATH_FULL)
    return

    ##################################################

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Parse MuseScore", description = "Extract information about and from MuseScore files.")
    parser.add_argument("-m", "--metadata_mapping", type = str, default = METADATA_MAPPING, help = "Absolute filepath to metadata-to-data table")
    parser.add_argument("-o", "--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory")
    parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # ARGS AND CONSTANTS
    ##################################################

    # parse arguments
    args = parse_args()

    # constant filepaths
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    OUTPUT_FILEPATH_ALL = f"{args.output_dir}/all_files.csv"
    OUTPUT_FILEPATH_FULL = f"{args.output_dir}/{basename(args.output_dir)}_full.csv"

    # for getting metadata
    METADATA = pd.read_csv(filepath_or_buffer = args.metadata_mapping, sep = ",", header = 0, index_col = False)
    METADATA = {path : path_metadata if not pd.isna(path_metadata) else None for path, path_metadata in zip(METADATA["data_path"], METADATA["metadata_path"])}

    # for encoding and decoding
    encoding = model_remi.representation.get_encoding() # load the encoding
    indexer = model_remi.representation.Indexer(data = encoding["event_code_map"])# get the indexer
    vocabulary = utils.inverse_dict(indexer.get_dict()) # for decoding

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    ##################################################

    # GET FULL LIST OF MUSESCORE FILES
    ##################################################

    # use glob to get all mscz files
    paths = glob.iglob(pathname = f"{MUSESCORE_DIR}/data/**", recursive = True) # glob filepaths recursively, generating an iterator object
    paths = tuple(path for path in paths if isfile(path) and path.endswith("mscz")) # filter out non-file elements that were globbed
    if exists(OUTPUT_FILEPATH_FULL):
        completed_paths = set(pd.read_csv(filepath_or_buffer = OUTPUT_FILEPATH_FULL, sep = ",", header = 0, index_col = False)["path"].tolist())
        paths = list(path for path in tqdm(iterable = paths, desc = "Determining Already-Complete Paths") if path not in completed_paths)
        paths = random.sample(population = paths, k = len(paths))

    ##################################################


    # GO THROUGH MUSESCORE FILES AND DETERMINE WHICH ARE PUBLIC DOMAIN
    ##################################################

    # use multiprocessing
    logging.info(f"N_PATHS = {len(paths):,}") # print number of paths to process
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = list(tqdm(iterable = pool.imap_unordered(func = get_full_dataset,
                                                           iterable = paths,
                                                           chunksize = CHUNK_SIZE),
                            desc = "Determining Full Dataset",
                            total = len(paths)))
    
    ##################################################

##################################################


# EXAMPLE METADATA FILE
##################################################
# {'data': {'as_pro': False,
#           'blocker_info': None,
#           'comments': {'comments': [], 'comments_total': 0},
#           'complexity': 1,
#           'composer': {'featured': 0,
#                        'id': 757,
#                        'name': 'W.A. Mozart',
#                        'uri': 'w.a._mozart',
#                        'url': 'https://musescore.com/sheetmusic/artists/w.a._mozart'},
#           'copyright_details': [],
#           'count_comments': 0,
#           'count_favorites': 1,
#           'count_views': 87,
#           'disable_hidden_url': '/score/manage/hidden/admin/unhide?score_id=6219296',
#           'dispute_hidden': '/contact?subject=An+appeal+against+the+hiding+of+my+score&message=https%3A%2F%2Fmusescore.com%2Fuser%2F30223591%2Fscores%2F6219296',
#           'error_description': None,
#           'genres': [{'name': 'classical',
#                       'url_to_search': '/sheetmusic/classical'}],
#           'groups': [{'title': 'Piano',
#                       'url': 'https://musescore.com/groups/piano/'}],
#           'hidden': False,
#           'isAddedToFavorite': False,
#           'isAddedToSpotlight': False,
#           'is_author_blocked_you': False,
#           'is_banned_user': False,
#           'is_blocked': False,
#           'is_can_rate_score': True,
#           'is_download_limited': False,
#           'is_ogg_supported': False,
#           'is_public_domain': True,
#           'is_similar_scores_more': True,
#           'is_user_follow': False,
#           'is_waiting_for_moderate': False,
#           'license_string': '<a '
#                             'href="https://creativecommons.org/publicdomain/zero/1.0/" '
#                             'target="_blank"><i class="icon-cc-zero"></i> '
#                             'Creative Commons copyright waiver</a>',
#           'license_url': 'https://creativecommons.org/publicdomain/zero/1.0/',
#           'limit_download_count': 20,
#           'official_score': None,
#           'opened_dispute': False,
#           'payment_account_id': None,
#           'paywall': {'has_instant_paywall': False,
#                       'is_trial_user': False,
#                       'period': None,
#                       'provider_name': None},
#           'pr_show': False,
#           'privacy_string': '<i class="icon-public"></i> Everyone can see this '
#                             'score',
#           'private_link_secret': None,
#           'score': {'_links': {'self': {'href': 'https://musescore.com/user/30223591/scores/6219296'}},
#                     'artist_name': 'Wolfgang Amadeus Mozart',
#                     'body': '',
#                     'can_manage_score': False,
#                     'comments_count': 0,
#                     'complexity': 1,
#                     'composer_name': 'W.A. Mozart',
#                     'date_created': 1592866549,
#                     'date_updated': 1621206702,
#                     'description': 'Solo',
#                     'duration': '00:42',
#                     'favorite_count': 1,
#                     'file_score_title': 'Contredanse in F',
#                     'has_custom_audio': False,
#                     'has_custom_video': False,
#                     'hits': 87,
#                     'id': 6219296,
#                     'instrumentation_id': 114,
#                     'instrumentations': [{'id': 114,
#                                           'is_active': 1,
#                                           'is_auto': 1,
#                                           'name': 'Solo',
#                                           'parent_id': 0,
#                                           'uri': 'solo',
#                                           'url_to_search': '/sheetmusic/solo',
#                                           'weight': 0}],
#                     'instruments': [{'count': 1,
#                                      'name': 'Piano',
#                                      'url_to_search': '/sheetmusic/piano'}],
#                     'is_blocked': False,
#                     'is_downloadable': 1,
#                     'is_draft': False,
#                     'is_official': False,
#                     'is_origin': False,
#                     'is_original': False,
#                     'is_private': 0,
#                     'is_public_domain': True,
#                     'keysig': 'F major, D minor',
#                     'license': 'cc-zero',
#                     'license_id': 4,
#                     'license_version': '4.0',
#                     'measures': 24,
#                     'pages_count': 1,
#                     'parts': 1,
#                     'parts_names': ['Piano'],
#                     'processing': 'ready',
#                     'publisher': None,
#                     'rating': {'abusive_ban_time': None,
#                                'abusive_ban_time_remain': None,
#                                'count': 0,
#                                'count_to_visible': 1,
#                                'rating': 0,
#                                'stats': [],
#                                'user_rating': None},
#                     'revision_id': 11323665,
#                     'revisions_count': 1,
#                     'share': {'embedUrl': 'https://musescore.com/user/30223591/scores/6219296/embed',
#                               'isShowSecretUrl': False,
#                               'publicUrl': 'https://musescore.com/user/30223591/scores/6219296',
#                               'title': 'W.A.+Mozart+-+Contredanse+in+F%2C+K.+15h',
#                               'url': 'https%3A%2F%2Fmusescore.com%2Fuser%2F30223591%2Fscores%2F6219296'},
#                     'song_name': 'The London Sketchbook, 15a - 15ss',
#                     'subtitle': 'K. 15h',
#                     'tags': ['Mozart', 'Contredanse', 'K 15h', 'Piano'],
#                     'thumbnails': {'large': 'https://musescore.com/static/musescore/scoredata/g/2ad4e34b891ad6712e0bb24eea409fde195f8fc9/score_0.png@500x660?no-cache=1621206702&bgclr=ffffff',
#                                    'medium': 'https://musescore.com/static/musescore/scoredata/g/2ad4e34b891ad6712e0bb24eea409fde195f8fc9/score_0.png@300x420?no-cache=1621206702&bgclr=ffffff',
#                                    'original': 'https://musescore.com/static/musescore/scoredata/g/2ad4e34b891ad6712e0bb24eea409fde195f8fc9/score_0.png?no-cache=1621206702',
#                                    'small': 'https://musescore.com/static/musescore/scoredata/g/2ad4e34b891ad6712e0bb24eea409fde195f8fc9/score_0.png@180x252?no-cache=1621206702&bgclr=ffffff'},
#                     'title': 'W.A. Mozart - Contredanse in F, K. 15h',
#                     'truncated_description': '',
#                     'url': 'https://musescore.com/user/30223591/scores/6219296',
#                     'user': {'_links': {'self': {'href': 'https://musescore.com/leandro15'}},
#                              'cover_url': 'https://musescore.com/static/musescore/userdata/cover/b/5/2/30223591.jpg?cache=1593540065',
#                              'date_created': 1541567170,
#                              'id': 30223591,
#                              'image': 'https://musescore.com/static/musescore/userdata/avatar/2/c/8/30223591.jpg@150x150?cache=1643839570',
#                              'is_moderator': False,
#                              'is_pro': False,
#                              'is_publisher': False,
#                              'is_staff': False,
#                              'name': 'Leandro15',
#                              'url': 'https://musescore.com/leandro15'}},
#           'score_blocked_by_country': False,
#           'score_type': 'regular',
#           'score_user_count': 0,
#           'secret': None,
#           'sets': [{'title': 'Wolfgang Amadeus Mozart - The London Sketchbook '
#                              '(K. 15a - K. 15ss)',
#                     'url': '/user/30223591/sets/5103958'}],
#           'similar_scores': [{'id': '6239405',
#                               'instrumentations': [{'name': 'Solo'}],
#                               'instruments': [{'name': 'Piano'}],
#                               'publisher': None,
#                               'rating': {'count': '4', 'rating': '4.85'},
#                               'title': 'W.A. Mozart - Presto in B-flat, K. '
#                                        '15ll',
#                               'url': '/user/30223591/scores/6239405'},
#                              {'id': '6204473',
#                               'instrumentations': [{'name': 'Solo'}],
#                               'instruments': [{'name': 'Piano'}],
#                               'publisher': None,
#                               'rating': {'count': '3', 'rating': '4.83'},
#                               'title': 'W.A. Mozart - Fantasia (Prelude) in G, '
#                                        'K. 15g',
#                               'url': '/user/30223591/scores/6204473'},
#                              {'id': '6238350',
#                               'instrumentations': [{'name': 'Solo'}],
#                               'instruments': [{'name': 'Piano'}],
#                               'publisher': None,
#                               'rating': {'count': '3', 'rating': '4.83'},
#                               'title': 'W.A. Mozart - Rondo in F, K. 15hh',
#                               'url': '/user/30223591/scores/6238350'}],
#           'song': {'_links': {'self': {'href': 'https://musescore.com/song/the_london_sketchbook_15a_15ss-2377559'}},
#                    'artist': {'_links': {'self': {'href': 'https://musescore.com/artist/wolfgang_amadeus_mozart-31435'}},
#                               'id': 31435,
#                               'name': 'Wolfgang Amadeus Mozart'},
#                    'id': 2377559,
#                    'name': 'The London Sketchbook, 15a - 15ss'}},
#  'score': {'fileVersion': 302,
#            'id': 6219296,
#            'isAddedToFavorite': 0,
#            'isBlocked': False,
#            'isDownloadable': True,
#            'isPublic': True,
#            'pagination': [],
#            'partNames': ['Piano'],
#            'scoreOfTheDay': [],
#            'share': {'embedUrl': 'https://musescore.com/user/30223591/scores/6219296/embed',
#                      'isShowSecretUrl': False,
#                      'publicUrl': 'https://musescore.com/user/30223591/scores/6219296',
#                      'title': 'W.A.+Mozart+-+Contredanse+in+F%2C+K.+15h',
#                      'url': 'https%3A%2F%2Fmusescore.com%2Fuser%2F30223591%2Fscores%2F6219296'},
#            'statDelayedUrl': 'https://musescore.com/score/stats/delayed-view?score_id=6219296',
#            'statUrl': 'https://musescore.com/score/stats/view?score_id=6219296',
#            'title': 'W.A. Mozart - Contredanse in F, K. 15h',
#            'url': 'https://musescore.com/user/30223591/scores/6219296',
#            'user': {'id': 30223591,
#                     'name': 'Leandro15',
#                     'url': 'https://musescore.com/leandro15'}},
#  'status_code': 200}
##################################################
