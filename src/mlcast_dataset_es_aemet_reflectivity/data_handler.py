import json
import os
import tarfile
from pathlib import Path
from typing import List, Optional

import requests


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("AEMET_API_KEY")
    if not key:
        raise ValueError(
            "API key not found. "
            "Pass it as the api_key argument or define the AEMET_API_KEY environment variable."
        )
    return key


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_gauge_data(
    FECHAINI="20230522T000000",
    FECHAFIN="20230523T000000",
    AREA="-1",
    FILTROS="PREC>=0",
    out_dir="./GaugeData",
    api_key: Optional[str] = None,
):
    api_key = _get_api_key(api_key)
    out_dir = _ensure_dir(out_dir)

    url = (
        "http://bigdata.aemet.es/bigdatarest/api/bdaut/observacion/"
        f"fechaini/{FECHAINI}/fechafin/{FECHAFIN}/idemas/-1/nombre_estaciones/-1/"
        f"area_estaciones/{AREA}/parametros/PREC/filtros/{FILTROS}/"
        "formato_salida/csv/flag_calidad/-1/solicitados/-1"
    )
    print("URL:", url)

    querystring = {"api_key": api_key}
    headers = {"cache-control": "no-cache"}

    response = requests.get(url, headers=headers, params=querystring, verify=False, timeout=120)
    response.raise_for_status()

    data = json.loads(response.text)
    estado = data["estado"]

    if estado == 200:
        print("Success")
        url_datos = data["datos"]
        datos = requests.get(url_datos, headers=headers, params=querystring, verify=False, timeout=300)
        datos.raise_for_status()

        output_file = out_dir / "datos.csv"
        output_file.write_bytes(datos.content)
        print(f"Saved to: {output_file}")
        return output_file
    else:
        raise RuntimeError(f"Could not retrieve the data. Status code: {estado}")


def download_radar_data(
    FECHAINI="20240707T000000",
    FECHAFIN="20240708T000000",
    IMAGEN="RNN",
    CONFIGURACION="24HR_CAPPI,24HR_SRI",
    RADAR="MUR",
    FORMATO="netcdf",
    EPSG="4326",
    out_dir="./RadarData",
    tar_name="fich.tar",
    api_key: Optional[str] = None,
) -> List[Path]:
    """
    Download radar data from AEMET BigData, extract the tar file, and return the
    list of extracted files.
    """
    api_key = _get_api_key(api_key)
    out_dir = _ensure_dir(out_dir)

    url = (
        "http://bigdata.aemet.es/bigdatarest/api/bdim/radares/producto/"
        f"fechaini/{FECHAINI}/fechafin/{FECHAFIN}/radar/{RADAR}/imagen/{IMAGEN}/"
        f"configuracion/{CONFIGURACION}/formato_salida/{FORMATO}/epsg/{EPSG}"
    )
    print("URL:", url)

    querystring = {"api_key": api_key}
    headers = {"cache-control": "no-cache"}

    response = requests.get(url, headers=headers, params=querystring, verify=False, timeout=120)
    response.raise_for_status()

    print(response.text)
    data = json.loads(response.text)
    estado = data["estado"]

    if estado != 200:
        raise RuntimeError(f"Could not retrieve the data. Status code: {estado}")

    print("Success")
    url_datos = data["datos"]
    datos = requests.get(url_datos, headers=headers, params=querystring, verify=False, timeout=600)
    datos.raise_for_status()

    tar_path = out_dir / tar_name
    tar_path.write_bytes(datos.content)

    extracted_files: List[Path] = []
    with tarfile.open(tar_path) as my_tar:
        for member in my_tar.getmembers():
            if not member.isfile():
                continue

            member_path = out_dir / member.name
            member_path.parent.mkdir(parents=True, exist_ok=True)

            extracted = my_tar.extractfile(member)
            if extracted is None:
                continue

            content = extracted.read()
            member_path.write_bytes(content)
            extracted_files.append(member_path)
            print(f"Extracted: {member_path}")

    tar_path.unlink(missing_ok=True)
    print(f"Tar file extracted into {out_dir}")

    return extracted_files


def download_sat_images(
    FECHAINI="20210301T000000",
    FECHAFIN="20210301T010000",
    PRODUCTO="CMA",
    out_dir="./SatImages",
    tar_name="fich.tar",
    api_key: Optional[str] = None,
) -> List[Path]:
    api_key = _get_api_key(api_key)
    out_dir = _ensure_dir(out_dir)

    url = (
        "http://bigdata.aemet.es/bigdatarest/api/bdim/saf/"
        f"fechaini/{FECHAINI}/fechafin/{FECHAFIN}/producto/{PRODUCTO}/formato_salida/-1"
    )
    print("URL:", url)

    querystring = {"api_key": api_key}
    headers = {"cache-control": "no-cache"}

    response = requests.get(url, headers=headers, params=querystring, verify=False, timeout=120)
    response.raise_for_status()

    print(response.text)
    data = json.loads(response.text)
    estado = data["estado"]

    if estado != 200:
        raise RuntimeError(f"Could not retrieve the data. Status code: {estado}")

    print("Success")
    url_datos = data["datos"]
    datos = requests.get(url_datos, headers=headers, params=querystring, verify=False, timeout=600)
    datos.raise_for_status()

    tar_path = out_dir / tar_name
    tar_path.write_bytes(datos.content)

    extracted_files: List[Path] = []
    with tarfile.open(tar_path) as my_tar:
        for member in my_tar.getmembers():
            if not member.isfile():
                continue

            member_path = out_dir / member.name
            member_path.parent.mkdir(parents=True, exist_ok=True)

            extracted = my_tar.extractfile(member)
            if extracted is None:
                continue

            content = extracted.read()
            member_path.write_bytes(content)
            extracted_files.append(member_path)
            print(f"Extracted: {member_path}")

    tar_path.unlink(missing_ok=True)
    print(f"Tar file extracted into {out_dir}")

    return extracted_files


if __name__ == "__main__":
    download_radar_data()