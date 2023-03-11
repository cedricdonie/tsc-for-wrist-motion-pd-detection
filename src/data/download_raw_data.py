# Scripts to download raw data from synapse.org
# Author(s): Cedric Donie (cedricdonie@gmail.com)
#
# See also https://www.synapse.org/#!Synapse:syn20681023/wiki/594678

import argparse
import dotenv
import os
import synapseutils
import synapseclient
from pathlib import Path

from src.data.global_constants import _PATHS

dotenv.load_dotenv(dotenv.find_dotenv())


parser = argparse.ArgumentParser()
parser.add_argument(
    "output_directory",
    type=Path,
    help="Directory in which to place the downloaded files",
)


def download_table(client, synapse_id, output_directory):
    tables_directory = output_directory / "TABLES"
    tables_directory.mkdir(exist_ok=True, parents=True)
    entity = client.get(synapse_id)
    name = "tbl_" + entity.name.lower().replace(" ", "_") + ".csv"
    query = client.tableQuery(f"select * from {synapse_id}")
    query.asDataFrame().to_csv(tables_directory / name)


def download_all_tables(client, output_directory):
    ids = """syn20681939
syn20681938
syn20681937
syn20681936
syn20681935
syn20681934
syn20681933
syn20681932
syn20681931
syn20681895
syn20681894
syn20681893
syn20681892
syn20681891
syn20681035
syn20681034
syn20681033
syn20681032
syn20681031""".split()
    for synapse_id in ids:
        download_table(client, synapse_id, output_directory)


if __name__ == "__main__":
    args = parser.parse_args()

    syn = synapseclient.login(
        os.environ["SYNAPSE_USERNAME"], os.environ["SYNAPSE_PASSWORD"]
    )
    download_all_tables(syn, args.output_directory)
    print("Downloaded all tables")
    files = synapseutils.syncFromSynapse(syn, "syn20681023", path=args.output_directory)
