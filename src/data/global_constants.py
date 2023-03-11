# Global Constants
# Author(s): Neha Das (neha.das@tum.de), Cedric Donié (cedricdonie@gmail.com)

from attrdict import AttrDict
import pathlib

_PATHS = AttrDict()

_PATHS.root = (
    pathlib.Path(__file__).parent.parent.parent.resolve() / "data" / "external"
)
_PATHS.processed_data = _PATHS.root.parent / "processed"

_PATHS.shimmer_back_folder_name = "Shimmer_Back"
_PATHS.shimmer_leftankle_folder_name = "Shimmer_LeftAnkle"
_PATHS.shimmer_rightankle_folder_name = "Shimmer_RightAnkle"
_PATHS.shimmer_leftwrist_folder_name = "Shimmer_LeftWrist"
_PATHS.shimmer_rightwrist_folder_name = "Shimmer_RightWrist"

_PATHS.geneactiv_folder_name = "GENEActiv"
_PATHS.pebble_folder_name = "Pebble"
_PATHS.phone_folder_name = "Phone"

_PATHS.smartdevice_task_tbl_path = "tbl_task_scores_-_part_i.csv"
_PATHS.shimmer_task_tbl_path = "tbl_task_scores_-_part_ii.csv"

_PATHS.action_dict_path = "tbl_task_code_dictionary.csv"

_PATHS.med_path = "tbl_medication_diary.csv"
_PATHS.home_tasks_path = "tbl_home_tasks.csv"
_PATHS.metadata_lab_visits_path = "tbl_metadata_of_laboratory_visits.csv"
_PATHS.metadata_lab_visits_dict_path = (
    "tbl_metadata_of_laboratory_visits_dictionary.csv"
)
_PATHS.sensor_grp_1_details_path = "tbl_sensor_data_-_part_i.csv"
_PATHS.sensor_grp_2_details_path = "tbl_sensor_data_-_part_ii.csv"
_PATHS.slp_diary_path = "tbl_sleep_diary.csv"
_PATHS.sub_diary_path = "tbl_subject_diary.csv"
_PATHS.sub_diary_dict_path = "tbl_subject_diary_dictionary.csv"
_PATHS.updrs_path = "tbl_updrs_responses.csv"
_PATHS.gcp_project_id = "ldopa-351322"
_PATHS.gcp_bucket_uri = f"gs://{_PATHS.gcp_project_id}-aiplatform"

_LISTS = AttrDict()

_LISTS.shimmer_locations = [
    _PATHS.shimmer_back_folder_name,
    _PATHS.shimmer_leftankle_folder_name,
    _PATHS.shimmer_rightankle_folder_name,
    _PATHS.shimmer_leftwrist_folder_name,
    _PATHS.shimmer_rightwrist_folder_name,
]

_LISTS.shimmer_patients = (
    "3_BOS",
    "4_BOS",
    "5_BOS",
    "6_BOS",
    "7_BOS",
    "8_BOS",
    "9_BOS",
    "10_BOS",
    "11_BOS",
    "12_BOS",
    "13_BOS",
    "14_BOS",
    "15_BOS",
    "16_BOS",
    "17_BOS",
    "18_BOS",
    "19_BOS",
)
_LISTS.day_list = ["1", "2", "3", "4"]

# Exclude "4_BOS" from training since the data point may be erroneous.
# See https://www.synapse.org/#!Synapse:syn20681023/discussion/threadId=9167
SUBJECTS = dict()
SUBJECTS["train"] = (
    "10_BOS",
    "10_NYC",
    "11_NYC",
    "12_BOS",
    "12_NYC",
    "13_BOS",
    "14_BOS",
    "16_BOS",
    "17_BOS",
    "18_BOS",
    "2_NYC",
    "3_BOS",
    "4_NYC",
    "5_NYC",
    "6_BOS",
    "6_NYC",
    "7_BOS",
    "7_NYC",
    "9_NYC",
)
SUBJECTS["val"] = ("15_BOS", "19_BOS")
SUBJECTS["test"] = ("11_BOS", "3_NYC", "5_BOS", "8_BOS", "8_NYC", "9_BOS")

_CONSTS = AttrDict()

_CONSTS.data_sampling_rate = 50
_CONSTS.seed = 6
