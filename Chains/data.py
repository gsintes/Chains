import toml
import sqlite3
import pathlib
import os
import shutil
from typing import Dict, List

import numpy as np


class Configuration():
    """Class to read and write configuration files compatible with FastTrack.

    """

    def read_toml(self, path: str) -> Dict[str, int]:
        """Read a configuration file from text file.

        Parameters
        ----------
        path : str
            Path pointing to the toml file.

        Returns
        -------
        Dict
            Parameters.

        """
        self.params = toml.load(path)
        return self.params["parameters"]


    def read_db(self, path: str) -> Dict[str, int]:
        """Read a configuration file from database.

        Parameters
        ----------
        path : str
            Path pointing to the sqlite database.

        Returns
        -------
        Dict
            Parameters.

        """
        cnx = sqlite3.connect(pathlib.Path(
            os.path.abspath(path)).as_uri() + "?mode=ro", uri=True)
        query = cnx.execute("SELECT parameter, value FROM parameter;")
        self.params = dict()
        for param, value in query:
            self.params[param] = value
        cnx.close()
        return self.params


    def write_toml(self, path: str) -> None:
        """Write a configuration file.

        Parameters
        ----------
        path : str
            Path pointing to the toml file.

        """
        with open(path, 'w') as f:
            toml.dump(self.params, f)


    def get_key(self, key: str) -> int:
        """Get a parameter from its key.

        Parameters
        ----------
        key : str
            Key.

        Returns
        -------
        Any
            Parameter.

        """
        return self.params["parameters"][key]

    def get_keys(self, keys: str) -> List[int]:
        """Get parameters from their keys.

        Parameters
        ----------
        keys : list
            List of keys.

        Returns
        -------
        List
            Parameters.

        """
        return [self.params["parameters"][key] for key in keys]


class Result():
    """Class to write result files compatible with FastTrack.

    """

    def add_data(self, dat) -> None:
        """Append data in the database.

            Parameters
            ----------
            dat : dict or list of dicts
                Data.

        """
        cursor = self.cnx.cursor()
        if not isinstance(dat, list):
            dat = [dat]
        for data in dat:
            cursor.execute("INSERT INTO tracking (xBody, yBody, tBody, areaBody,"
                           "perimeterBody, bodyMajorAxisLength, bodyMinorAxisLength, bodyExcentricity,"
                           "imageNumber, id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ",
                           (data["xcenter"], data["ycenter"], data["orientation"],
                            data["area"], data["perim"],
                            data["major_axis"], data["minor_axis"],
                            np.sqrt(1 - (data["minor_axis"] / data["major_axis"]) ** 2),
                            data["time"], data["id"]))
        self.cnx.commit()
        cursor.close()

    def __init__(self, path: str) -> None:
        path = os.path.abspath(path + "/Tracking_Result/")
        try:
            os.makedirs(path)
        except FileExistsError:
            try:
                os.remove(os.path.join(path, "tracking.db"))
            except FileNotFoundError:
                pass
        self.cnx = sqlite3.connect(os.path.join(path,"tracking.db"))
        cursor = self.cnx.cursor()
        cursor.execute("CREATE TABLE tracking (xBody REAL, yBody REAL, tBody REAL,"
                       "areaBody REAL, perimeterBody REAL,"
                       "bodyMajorAxisLength REAL, bodyMinorAxisLength REAL,"
                       "bodyExcentricity REAL, imageNumber INTEGER, id INTEGER)")
        self.cnx.commit()
        cursor.close()

    def __del__(self) -> None:
        self.cnx.close()
