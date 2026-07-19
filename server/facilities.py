"""
Facility onboarding: industry-type templates and facility creation. This is what turns
the multi-tenant schema into an actual product flow -- at login you either enter the
seeded demo plant (benchmark scenarios, trained GNN, full guided demo) or create your
own facility from an industry template (zones pre-laid-out on the plant-map canvas,
labels editable), and every dashboard page then scopes to that facility via the
facility_id query param the routers already accept.

A created facility starts as a live shell: zone map, device hub, IoT ingest, alerts,
and the RAG chat all work immediately; the benchmark scenarios/GNN demo remain
exclusive to the demo plant because they're tied to the TEP simulator's 7 zones.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.models import Facility, Zone
from db.session import get_db

router = APIRouter(tags=["facilities"])

# Zone boxes are on the same 860x360 canvas as frontend/src/PlantMap.jsx. Each template
# is a realistic monitored-area set for that industry; control rooms get
# has_sensors=False (no physical process hazard), matching the demo plant's convention.
FACILITY_TEMPLATES = {
    "chemical_plant": {
        "label": "Chemical / process plant",
        "description": "Continuous process units connected by process flow (the demo plant's layout).",
        "zones": [
            {"key": "feed_zone", "label": "Feed Section", "x": 24, "y": 150, "w": 140, "h": 92},
            {"key": "reactor_zone", "label": "Reactor", "x": 196, "y": 134, "w": 140, "h": 118},
            {"key": "separator_zone", "label": "Separator", "x": 368, "y": 150, "w": 140, "h": 92},
            {"key": "stripper_zone", "label": "Stripper", "x": 540, "y": 150, "w": 130, "h": 92},
            {"key": "condenser_zone", "label": "Condenser", "x": 368, "y": 32, "w": 140, "h": 80},
            {"key": "compressor_zone", "label": "Recycle Compressor", "x": 184, "y": 32, "w": 164, "h": 80},
            {"key": "control_room", "label": "Control Room", "x": 700, "y": 32, "w": 130, "h": 80, "has_sensors": False},
        ],
    },
    "steel_plant": {
        "label": "Steel plant",
        "description": "Coke ovens through casting and rolling -- high-temperature and molten-metal areas.",
        "zones": [
            {"key": "coke_oven", "label": "Coke Oven Battery", "x": 24, "y": 32, "w": 150, "h": 80},
            {"key": "blast_furnace", "label": "Blast Furnace", "x": 204, "y": 32, "w": 150, "h": 80},
            {"key": "steel_melt_shop", "label": "Steel Melt Shop", "x": 384, "y": 32, "w": 150, "h": 80},
            {"key": "casting_bay", "label": "Continuous Casting Bay", "x": 114, "y": 150, "w": 160, "h": 92},
            {"key": "rolling_mill", "label": "Rolling Mill", "x": 304, "y": 150, "w": 150, "h": 92},
            {"key": "control_room", "label": "Control Room", "x": 700, "y": 32, "w": 130, "h": 80, "has_sensors": False},
        ],
    },
    "refinery": {
        "label": "Oil refinery",
        "description": "Crude distillation, conversion units, tankage and flare systems.",
        "zones": [
            {"key": "crude_unit", "label": "Crude Distillation Unit", "x": 24, "y": 32, "w": 160, "h": 80},
            {"key": "fcc_unit", "label": "FCC Unit", "x": 214, "y": 32, "w": 140, "h": 80},
            {"key": "hydrotreater", "label": "Hydrotreater", "x": 384, "y": 32, "w": 140, "h": 80},
            {"key": "tank_farm", "label": "Tank Farm", "x": 114, "y": 150, "w": 160, "h": 92},
            {"key": "flare_area", "label": "Flare & Relief Area", "x": 304, "y": 150, "w": 150, "h": 92},
            {"key": "control_room", "label": "Control Room", "x": 700, "y": 32, "w": 130, "h": 80, "has_sensors": False},
        ],
    },
    "pharma_plant": {
        "label": "Pharmaceutical plant",
        "description": "API synthesis, clean rooms, and solvent handling.",
        "zones": [
            {"key": "api_synthesis", "label": "API Synthesis Block", "x": 24, "y": 32, "w": 160, "h": 80},
            {"key": "clean_room", "label": "Clean Room Suite", "x": 214, "y": 32, "w": 150, "h": 80},
            {"key": "solvent_store", "label": "Solvent Storage", "x": 394, "y": 32, "w": 140, "h": 80},
            {"key": "formulation", "label": "Formulation & Packaging", "x": 114, "y": 150, "w": 180, "h": 92},
            {"key": "utilities", "label": "Utilities & Boilers", "x": 324, "y": 150, "w": 150, "h": 92},
            {"key": "control_room", "label": "Control Room", "x": 700, "y": 32, "w": 130, "h": 80, "has_sensors": False},
        ],
    },
    "mining_mineral": {
        "label": "Mining / mineral processing",
        "description": "Crushing, grinding, and flotation circuits with conveyor galleries.",
        "zones": [
            {"key": "crusher_house", "label": "Crusher House", "x": 24, "y": 32, "w": 150, "h": 80},
            {"key": "conveyor_gallery", "label": "Conveyor Gallery", "x": 204, "y": 32, "w": 160, "h": 80},
            {"key": "ball_mill", "label": "Grinding Mill", "x": 394, "y": 32, "w": 140, "h": 80},
            {"key": "flotation_cells", "label": "Flotation Cells", "x": 114, "y": 150, "w": 160, "h": 92},
            {"key": "tailings", "label": "Tailings Handling", "x": 304, "y": 150, "w": 150, "h": 92},
            {"key": "control_room", "label": "Control Room", "x": 700, "y": 32, "w": 130, "h": 80, "has_sensors": False},
        ],
    },
}


class ZoneSpec(BaseModel):
    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=200)
    x: float = 0
    y: float = 0
    w: float = 100
    h: float = 100
    has_sensors: bool = True


class FacilityCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    location: str | None = Field(default=None, max_length=200)
    industry_type: str = Field(min_length=1, max_length=50)
    zones: list[ZoneSpec] = Field(min_length=1, max_length=20)


@router.get("/facility-templates")
def get_facility_templates():
    return [{"industry_type": key, **tpl} for key, tpl in FACILITY_TEMPLATES.items()]


@router.post("/facilities", status_code=201)
def create_facility(body: FacilityCreate, db: Session = Depends(get_db)):
    if db.query(Facility).filter_by(name=body.name).first() is not None:
        raise HTTPException(status_code=409, detail=f"A facility named '{body.name}' already exists")
    keys = [z.key for z in body.zones]
    if len(keys) != len(set(keys)):
        raise HTTPException(status_code=422, detail="Zone keys must be unique")

    facility = Facility(name=body.name, location=body.location, industry_type=body.industry_type)
    db.add(facility)
    db.flush()  # get facility.id for the zone FKs
    for z in body.zones:
        db.add(Zone(facility_id=facility.id, key=z.key, label=z.label,
                    layout_x=z.x, layout_y=z.y, layout_w=z.w, layout_h=z.h,
                    has_sensors=z.has_sensors))
    db.commit()
    return {"id": str(facility.id), "name": facility.name, "location": facility.location,
            "industry_type": facility.industry_type, "n_zones": len(body.zones)}
