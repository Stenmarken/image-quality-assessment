import numpy as np
from utils import filter_by_key, filter_entries


def filter_entries(data, **filters):
    return [
        entry for entry in data if all(entry.get(k) == v for k, v in filters.items())
    ]


def dict_lists_equal(list1, list2):
    set1 = {tuple(sorted(d.items())) for d in list1}
    set2 = {tuple(sorted(d.items())) for d in list2}
    return set1 == set2


def test_log_location_wise(results):
    total = filter_by_key(results, "SRCC")
    otaniemi = filter_by_key(filter_entries(results, location="Otaniemi"), "SRCC")
    munkkivuori = filter_by_key(filter_entries(results, location="Munkkivuori"), "SRCC")
    assert np.isclose(np.mean(total), np.mean([otaniemi, munkkivuori]))


def test_log_weather_wise(results):
    total = filter_by_key(results, "SRCC")
    all_rainy = filter_by_key(filter_entries(results, weather="rainy"), "SRCC")
    fst_foggy = filter_by_key(
        filter_entries(results, weather="foggy_0.7_0.0_1.0"), "SRCC"
    )
    snd_foggy = filter_by_key(filter_entries(results, weather="foggy_0.5_0.9"), "SRCC")
    assert np.isclose(np.mean(total), np.mean([all_rainy, fst_foggy, snd_foggy]))


def test_mean_munkkivuori(results):
    munkkivuori = filter_by_key(filter_entries(results, location="Munkkivuori"), "SRCC")
    munkkivuori_foggy_fst = filter_by_key(
        filter_entries(results, location="Munkkivuori", weather="foggy_0.7_0.0_1.0"),
        "SRCC",
    )
    munkkivuori_foggy_snd = filter_by_key(
        filter_entries(results, location="Munkkivuori", weather="foggy_0.5_0.9"), "SRCC"
    )
    munkkivuori_rain = filter_by_key(
        filter_entries(results, location="Munkkivuori", weather="rainy"), "SRCC"
    )
    assert np.isclose(
        np.mean(munkkivuori),
        np.mean([munkkivuori_foggy_fst, munkkivuori_foggy_snd, munkkivuori_rain]),
    )


def test_mean_otaniemi(results):
    otaniemi = filter_by_key(filter_entries(results, location="Otaniemi"), "SRCC")
    otaniemi_foggy_fst = filter_by_key(
        filter_entries(results, location="Otaniemi", weather="foggy_0.7_0.0_1.0"),
        "SRCC",
    )
    otaniemi_foggy_snd = filter_by_key(
        filter_entries(results, location="Otaniemi", weather="foggy_0.5_0.9"), "SRCC"
    )

    otaniemi_rain = filter_by_key(
        filter_entries(results, location="Otaniemi", weather="rainy"), "SRCC"
    )

    assert np.isclose(
        np.mean(otaniemi),
        np.mean([otaniemi_foggy_fst, otaniemi_foggy_snd, otaniemi_rain]),
    )
