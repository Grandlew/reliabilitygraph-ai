from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Response,
    status,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .adjudication import (
    AdjudicationService,
    ReviewWorkflowError,
    ReviewerRole,
)
from .contracts import IncidentAdjudication


@dataclass(frozen=True)
class ReviewPrincipal:
    reviewer_pseudonym: str
    role: ReviewerRole


class TokenAuthorizer:
    """Minimal pilot adapter; production should supply an IdP-backed adapter."""

    def __init__(
        self,
        tokens: dict[str, ReviewPrincipal],
    ) -> None:
        if not tokens:
            raise ValueError("At least one review access token is required")
        self._tokens = dict(tokens)

    def authenticate(
        self,
        authorization: Annotated[str | None, Header()] = None,
    ) -> ReviewPrincipal:
        if (
            authorization is None
            or not authorization.startswith("Bearer ")
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer authentication required",
            )
        supplied = authorization.removeprefix("Bearer ").strip()
        for token, principal in self._tokens.items():
            if hmac.compare_digest(supplied, token):
                return principal
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid review credential",
        )


class AdjudicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adjudication: IncidentAdjudication


class ResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    final_adjudication: IncidentAdjudication
    disagreement_reason: str = Field(min_length=1, max_length=4000)


CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>NRIM shadow review</title>
  <style>
    body{font:16px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}
    label{display:block;margin-top:1rem;font-weight:600}
    input,textarea,button{font:inherit} input,textarea{box-sizing:border-box;width:100%;padding:.5rem}
    textarea{min-height:12rem;font-family:ui-monospace,monospace}
    button{margin:.75rem .5rem 0 0;padding:.5rem .8rem}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1rem}
    .warning{border-left:.3rem solid #b65d00;padding-left:1rem}
  </style>
  <script src="/nrim-shadow/review/console.js" defer></script>
</head>
<body>
  <h1>NRIM shadow evidence review</h1>
  <p class="warning">Complete and submit the blinded initial assessment before
  revealing NRIM evidence. Tokens are kept only in this page's memory and are
  not persisted by the console.</p>
  <label for="token">Bearer token</label>
  <input id="token" type="password" autocomplete="off">
  <label for="case-id">Case ID</label>
  <input id="case-id" autocomplete="off">
  <button id="status">Load workflow status</button>
  <button id="reveal">Reveal NRIM evidence</button>
  <h2>Adjudication payload</h2>
  <p>Paste one strict <code>IncidentAdjudication</code> JSON object. Initial
  submissions require <code>blinded_initial_assessment: true</code>;
  post-reveal submissions require <code>false</code>.</p>
  <label for="adjudication">Adjudication JSON</label>
  <textarea id="adjudication" spellcheck="false"></textarea>
  <button id="initial">Submit blinded initial review</button>
  <button id="post-reveal">Submit post-reveal review</button>
  <h2>Disagreement resolution</h2>
  <label for="reason">Resolution reason</label>
  <textarea id="reason"></textarea>
  <button id="resolve">Record resolution using adjudication above</button>
  <h2>Workflow/evidence output</h2>
  <pre id="output" aria-live="polite">No evidence loaded.</pre>
</body>
</html>
"""


CONSOLE_JAVASCRIPT = """(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const output = byId("output");
  const endpoint = (suffix) => {
    const caseId = encodeURIComponent(byId("case-id").value.trim());
    if (!caseId) throw new Error("Case ID is required.");
    return `/nrim-shadow/review/cases/${caseId}/${suffix}`;
  };
  const request = async (suffix, method = "GET", body = undefined) => {
    const token = byId("token").value;
    if (!token) throw new Error("Bearer token is required.");
    const options = {
      method,
      credentials: "same-origin",
      cache: "no-store",
      headers: {"Authorization": `Bearer ${token}`}
    };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(endpoint(suffix), options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(`${response.status}: ${JSON.stringify(payload)}`);
    }
    output.textContent = JSON.stringify(payload, null, 2);
  };
  const adjudication = () => {
    const value = JSON.parse(byId("adjudication").value);
    if (value === null || Array.isArray(value) || typeof value !== "object") {
      throw new Error("Adjudication JSON must be an object.");
    }
    return value;
  };
  const act = (operation) => async () => {
    output.textContent = "Working...";
    try { await operation(); }
    catch (error) { output.textContent = `Error: ${error.message}`; }
  };
  byId("status").addEventListener("click", act(() => request("status")));
  byId("reveal").addEventListener("click", act(() => request("reveal", "POST")));
  byId("initial").addEventListener("click", act(() =>
    request("initial", "POST", {adjudication: adjudication()})));
  byId("post-reveal").addEventListener("click", act(() =>
    request("post-reveal", "POST", {adjudication: adjudication()})));
  byId("resolve").addEventListener("click", act(() =>
    request("resolve", "POST", {
      final_adjudication: adjudication(),
      disagreement_reason: byId("reason").value
    })));
})();"""


def _console_security_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "Content-Security-Policy": (
            "default-src 'none'; script-src 'self'; "
            "style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'none'; object-src 'none'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'"
        ),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }


def create_review_router(
    *,
    service: AdjudicationService,
    authorizer: TokenAuthorizer,
) -> APIRouter:
    router = APIRouter(prefix="/nrim-shadow/review", tags=["nrim-shadow"])
    principal_dependency = Depends(authorizer.authenticate)

    @router.get(
        "/console",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def console() -> HTMLResponse:
        return HTMLResponse(
            CONSOLE_HTML,
            headers=_console_security_headers(),
        )

    @router.get("/console.js", include_in_schema=False)
    def console_javascript() -> Response:
        return Response(
            CONSOLE_JAVASCRIPT,
            media_type="text/javascript",
            headers=_console_security_headers(),
        )

    def translate(error: ReviewWorkflowError) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )

    @router.get("/cases/{case_id}/status")
    def case_status(
        case_id: str,
        principal: ReviewPrincipal = principal_dependency,
    ) -> dict:
        del principal
        try:
            return service.workflow_status(case_id)
        except ReviewWorkflowError as error:
            raise translate(error) from error

    @router.post("/cases/{case_id}/initial", status_code=201)
    def submit_initial(
        case_id: str,
        request: AdjudicationRequest,
        principal: ReviewPrincipal = principal_dependency,
    ) -> dict:
        if principal.role not in {
            ReviewerRole.REVIEWER,
            ReviewerRole.RESOLVER,
        }:
            raise HTTPException(status_code=403, detail="Reviewer role required")
        if (
            request.adjudication.reviewer_id_pseudonym
            != principal.reviewer_pseudonym
        ):
            raise HTTPException(
                status_code=403,
                detail="Reviewer identity differs from authenticated principal",
            )
        try:
            service.submit_initial(
                case_id=case_id,
                adjudication=request.adjudication,
            )
        except ReviewWorkflowError as error:
            raise translate(error) from error
        return {"status": "recorded"}

    @router.post("/cases/{case_id}/reveal")
    def reveal(
        case_id: str,
        principal: ReviewPrincipal = principal_dependency,
    ) -> dict:
        try:
            evidence = service.reveal_nrim(
                case_id=case_id,
                reviewer_pseudonym=principal.reviewer_pseudonym,
            )
        except ReviewWorkflowError as error:
            raise translate(error) from error
        return {"predictions": evidence}

    @router.post("/cases/{case_id}/post-reveal", status_code=201)
    def submit_post_reveal(
        case_id: str,
        request: AdjudicationRequest,
        principal: ReviewPrincipal = principal_dependency,
    ) -> dict:
        if (
            request.adjudication.reviewer_id_pseudonym
            != principal.reviewer_pseudonym
        ):
            raise HTTPException(status_code=403, detail="Reviewer mismatch")
        try:
            service.submit_post_reveal(
                case_id=case_id,
                adjudication=request.adjudication,
            )
        except ReviewWorkflowError as error:
            raise translate(error) from error
        return {"status": "recorded"}

    @router.post("/cases/{case_id}/resolve", status_code=201)
    def resolve(
        case_id: str,
        request: ResolutionRequest,
        principal: ReviewPrincipal = principal_dependency,
    ) -> dict:
        if principal.role is not ReviewerRole.RESOLVER:
            raise HTTPException(status_code=403, detail="Resolver role required")
        try:
            service.record_resolution(
                case_id=case_id,
                resolver_pseudonym=principal.reviewer_pseudonym,
                final_adjudication=request.final_adjudication,
                disagreement_reason=request.disagreement_reason,
            )
        except ReviewWorkflowError as error:
            raise translate(error) from error
        return {"status": "resolved"}

    return router
