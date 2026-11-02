"""Scenario lab API. Simulation compute: served locally only, never in the public build (see backend/web.py)."""
from fastapi import APIRouter, HTTPException

from backend.app.explorer import Explorer, IndexMissing
from backend.app.lab import Lab, SensorFault, TwinRun

router = APIRouter(prefix="/lab", tags=["scenario-lab"])
lab = Lab(Explorer())


def _guard(call):
    try:
        return call()
    except IndexMissing as e:
        raise HTTPException(503, detail={"code": "explorer_index_missing", "build": "python -m scripts.build_explorer_index"}) from e
    except KeyError as e:
        raise HTTPException(404, detail={"code": "background_not_recorded"}) from e
    except ValueError as e:
        raise HTTPException(400, detail={"code": "window_invalid" if str(e) == "window" else "background_incomplete" if str(e) == "coverage" else "request_invalid"}) from e


@router.get("/backgrounds")
def backgrounds(hours: int = 12):
    if not 6 <= hours <= 24:
        raise HTTPException(400, detail={"code": "request_invalid"})
    return _guard(lambda: {"backgrounds": lab.backgrounds(hours)})


@router.post("/sensor-fault")
def sensor_fault(request: SensorFault):
    return _guard(lambda: lab.sensor_fault(request))


@router.post("/twin")
def twin(request: TwinRun):
    return _guard(lambda: lab.twin(request))
