//! Shared client-side verifier for ChirpAuth-issued ID tokens.
//!
//! ChirpAuth (the sibling crate / binary in this workspace) mints RS256 JWTs
//! and publishes its public keys via a JWKS endpoint at `{issuer}/jwks.json`.
//! Consumers (Drive, Exchange, Social Graph / personal-ecosystem, …) all need
//! the same verification routine: parse the JWT, fetch JWKS, verify the
//! signature, check `iss`/`aud`/`exp`, and return the verified subject.
//!
//! ChirpAuth issues two shapes of token under the same JWKS:
//!
//! - **Human** tokens come out of the Authorization Code + PKCE flow. They
//!   may carry a `name` claim and have no `act` claim. They never carry
//!   `email` — ChirpAuth never hands a relying party a user's address.
//! - **Machine** tokens come out of the `client_credentials` grant for
//!   confidential clients. They carry `act: "machine"` and `owner_sub`
//!   (the human chirp-sub responsible for the client) and never carry
//!   `email`. See `protocols/specs/machine-identity.md`.
//!
//! Callers opt in to machine acceptance with [`VerifyOptions::accept_machine`]
//! (default `false`) and name the machine `client_id`s they will act on via
//! [`VerifyOptions::accepted_machine_audiences`] (fail-closed: an empty set
//! accepts no machine token). A machine token's `aud` is its own minting
//! client, not this relying party, so the human-path audience allowlist cannot
//! gate it; this set is the one place that membership rule lives — consumers no
//! longer hand-roll their own `client_id` allowlist. Services that only handle
//! humans don't change behavior; the returned [`ChirpVerifiedIdentity`] is an
//! enum, so call sites `match` on `Human { .. }` / `Machine { .. }`.
//!
//! [`verify_chirp_id_token`] returns a [`ChirpVerifiedToken`] carrying two
//! orthogonal axes: the [`Environment`] (`Production`/`Test`) the verifying
//! keyset belongs to — **provenance, derived from which issuer/JWKS matched,
//! not from any claim** — and the [`ChirpVerifiedIdentity`] principal, a sum
//! type TOTAL over the RP-visible classes (`Human` / `Machine` / `Test`). A
//! relying party therefore cannot mistake a test identity for a production
//! human, or a machine for a human, by forgetting a marker: every distinction
//! is in the type and a missed class is a compile error. Access the principal
//! via `token.identity`. (Two further service classes never surface here as
//! identities: a `dev_…`-key bearer fails key lookup, and a key-binding cert is
//! verified through [`verify_rs256_jws`], not as a principal — see
//! [`ChirpVerifiedIdentity`].)
//!
//! ## Environment enforcement (0.7)
//!
//! The environment axis is now **enforced**, not merely reported. The verifier
//! derives the expected [`Environment`] from the configured issuer
//! ([`ChirpAuthConfig::environment`]) and rejects any token whose provenance
//! disagrees:
//!
//! - A **production**-configured relying party rejects a token carrying
//!   `test == true` with [`ChirpAuthError::EnvironmentMismatch`].
//! - A **test**-configured relying party (issuer `…/test/{tenant}`) rejects a
//!   token that does *not* carry `test == true`, also with
//!   [`ChirpAuthError::EnvironmentMismatch`].
//!
//! Test-acceptance is therefore derived from the issuer's environment, not from
//! a per-call flag.

pub mod mint;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::signature::{RSA_PKCS1_2048_8192_SHA256, RsaPublicKeyComponents};
use serde::Deserialize;
use std::collections::BTreeSet;
use time::OffsetDateTime;

/// The canonical production ChirpAuth issuer.
///
/// Consumers should derive their [`ChirpAuthConfig`] from this rather than
/// re-typing the URL; a typo in an issuer string is a silent
/// [`ChirpAuthError::ClaimMismatch`] at runtime.
pub const DEFAULT_ISSUER: &str = "https://signin.chirpauth.com";

/// Endpoint and audience to verify ChirpAuth tokens against.
///
/// Construct via [`ChirpAuthConfig::new`] (or [`ChirpAuthConfig::from_env`]).
/// The `jwks_uri` is **derived** (`{issuer}/jwks.json`) and cannot be set
/// independently of the issuer — this removes a whole class of misconfiguration
/// where a relying party verifies issuer A's claims against issuer B's keyset.
/// Tests that need to point the fetch at an in-process server use
/// [`ChirpAuthConfig::with_jwks_uri`].
#[derive(Clone, Debug)]
pub struct ChirpAuthConfig {
    issuer: String,
    jwks_uri: String,
    /// The set of audiences this relying party accepts. A token is accepted if
    /// *any* of its `aud` values is in this set. Always non-empty.
    accepted_audiences: BTreeSet<String>,
}

impl ChirpAuthConfig {
    /// Build a config for a single audience.
    ///
    /// The issuer is normalized (trimmed, trailing slash stripped) and the
    /// `jwks_uri` is derived as `{issuer}/jwks.json`.
    pub fn new(issuer: impl Into<String>, audience: impl Into<String>) -> Self {
        Self::with_audiences(issuer, std::iter::once(audience.into()))
    }

    /// Build a config that accepts any of several audiences (an allowlist).
    ///
    /// A token is accepted when *any* of its `aud` values is in `audiences`,
    /// checked in a single pass — callers no longer loop a verify-per-audience.
    /// An empty `audiences` iterator yields a config that accepts nothing
    /// (every token fails `aud` matching with [`ChirpAuthError::ClaimMismatch`]).
    pub fn with_audiences(
        issuer: impl Into<String>,
        audiences: impl IntoIterator<Item = String>,
    ) -> Self {
        let issuer = normalize_issuer(issuer.into());
        let jwks_uri = format!("{issuer}/jwks.json");
        Self {
            issuer,
            jwks_uri,
            accepted_audiences: audiences.into_iter().map(|a| a.trim().to_owned()).collect(),
        }
    }

    /// Override the JWKS fetch URL. **Test-only**: production callers must let
    /// the URL derive from the issuer. Returns `self` for chaining.
    pub fn with_jwks_uri(mut self, jwks_uri: impl Into<String>) -> Self {
        self.jwks_uri = jwks_uri.into();
        self
    }

    /// Add another accepted audience to the allowlist.
    pub fn add_audience(mut self, audience: impl Into<String>) -> Self {
        self.accepted_audiences.insert(audience.into().trim().to_owned());
        self
    }

    /// The normalized issuer this config verifies against.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// The derived JWKS endpoint.
    pub fn jwks_uri(&self) -> &str {
        &self.jwks_uri
    }

    /// The audience allowlist.
    pub fn accepted_audiences(&self) -> &BTreeSet<String> {
        &self.accepted_audiences
    }

    /// The [`Environment`] this config's issuer belongs to — the provenance
    /// the verifier enforces. A production issuer yields
    /// [`Environment::Prod`]; a `…/test/{tenant}` issuer yields
    /// [`Environment::Test`].
    pub fn environment(&self) -> Environment {
        environment_from_issuer(&self.issuer)
    }

    /// Build a config from `{prefix}_ISSUER` and `{prefix}_AUDIENCE` env vars.
    ///
    /// Pass an empty `prefix` for unprefixed `CHIRP_AUTH_ISSUER` /
    /// `CHIRP_AUTH_AUDIENCE`. Returns `None` if either variable is missing or
    /// empty — the caller is expected to treat that as "ChirpAuth not
    /// configured" rather than an error.
    pub fn from_env(prefix: &str) -> Option<Self> {
        let (issuer_var, audience_var) = env_var_names(prefix);
        let issuer = std::env::var(&issuer_var)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())?;
        let audience = std::env::var(&audience_var)
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())?;
        Some(Self::new(issuer, audience))
    }
}

fn normalize_issuer(issuer: String) -> String {
    issuer.trim().trim_end_matches('/').to_owned()
}

fn env_var_names(prefix: &str) -> (String, String) {
    if prefix.is_empty() {
        ("CHIRP_AUTH_ISSUER".to_owned(), "CHIRP_AUTH_AUDIENCE".to_owned())
    } else {
        (format!("{prefix}_ISSUER"), format!("{prefix}_AUDIENCE"))
    }
}

/// The verified identity extracted from a ChirpAuth ID token — TOTAL over the
/// classes a relying party can legitimately receive through
/// [`verify_chirp_id_token`].
///
/// INV-09 (token-class ADT): this is the RP-facing projection of the service's
/// `VerifiedToken` sum type (`docs/inv-09-token-class-adt.md`). "Which class is
/// this?" is a property of the **type**, recovered once at verify and `match`ed
/// at every call site, rather than re-derived from which optional discriminator
/// (`act`/`test`) happens to be set. A relying party that handles only
/// production humans must add a `Test`/`Machine` arm or fail to compile — a
/// test principal can never be silently honored as a production human.
///
/// Dispatch is the triple `(typ family, issuer-derived environment, act)`:
///
/// - `typ == `[`ID_TOKEN_TYP`]: the ID-token family (Human/Machine/Test). A
///   key-binding cert (`typ == `[`KEYBIND_TYP`]) is rejected here and is NEVER
///   an identity — it is verified through [`verify_rs256_jws`] as a confirmation
///   artifact, not a principal, so a `cnf` cert cannot be confused with a login.
/// - production environment + `act` absent → [`ChirpVerifiedIdentity::Human`].
///   A human never carries `email` or `name` — both are structurally absent.
/// - `act == "machine"` → [`ChirpVerifiedIdentity::Machine`]. `owner_sub` is the
///   human chirp-sub responsible for the confidential client; `client_id` is the
///   token's `aud` claim (single-value).
/// - test environment + `act` absent → [`ChirpVerifiedIdentity::Test`]. Only a
///   test-issuer-configured RP can observe this (environment enforcement rejects
///   a `test` token at a production RP before any identity is built); it mirrors
///   the service's `VerifiedToken::Test`, keeping a test principal structurally
///   distinct from a production human.
///
/// Two RP-facing classes the service models but that DO NOT appear here, by
/// construction — so neither can masquerade as an identity:
/// - **DevBearer** (`dev_…`-key control-plane bearer): the dev signing key is
///   excluded from this verifier's keyset, so a dev token fails key lookup
///   ([`ChirpAuthError::NoMatchingKey`]) and is never returned as an identity.
/// - **KeyBind** (`chirp-keybind+jwt` confirmation cert): not an identity at all
///   — verified via [`verify_rs256_jws`] with `expected_typ = `[`KEYBIND_TYP`],
///   rejected on the ID-token path by the `typ` gate.
///
/// `sub` is always non-empty, ≤ 128 chars, and free of control characters.
#[derive(Clone, Debug)]
pub enum ChirpVerifiedIdentity {
    // NOTE: a Human identity is JUST the pairwise `sub`. There is intentionally
    // NO `email` and NO `name` — ChirpAuth never hands a relying party a
    // human-readable identifier (apps get a different opaque sub per app, by the
    // no-email/pairwise privacy design). The address/name live only in chirp,
    // used to send magic links / route mediated contact. See chirp-auth
    // docs/mediated-contact-north-star.md.
    Human { sub: String },
    Machine {
        sub: String,
        owner_sub: String,
        client_id: String,
    },
    /// A test-environment principal: a non-machine token verified against a
    /// `…/test/{tenant}` issuer's keyset (so its [`Environment`] is
    /// [`Environment::Test`]). Structurally distinct from [`Human`] so a
    /// production surface cannot honor a test identity by forgetting to inspect
    /// the environment axis — mirrors the service's `VerifiedToken::Test`. The
    /// keyset-provenance tenant lives on the [`ChirpVerifiedToken::environment`]
    /// axis (derived from the verified issuer), not duplicated here.
    ///
    /// [`Human`]: ChirpVerifiedIdentity::Human
    Test { sub: String },
}

impl ChirpVerifiedIdentity {
    /// The token's `sub` claim regardless of variant.
    pub fn sub(&self) -> &str {
        match self {
            Self::Human { sub, .. } | Self::Machine { sub, .. } | Self::Test { sub, .. } => sub,
        }
    }
}

/// Which ChirpAuth keyset verified this token — **provenance, not a claim**.
///
/// Derived from the issuer the token was verified against (and matched
/// exactly): a ChirpAuth test issuer is structurally `{prod_issuer}/test/{tenant}`
/// with its own per-tenant signing key and JWKS. A relying party configured with
/// only the production issuer therefore *cannot* observe [`Environment::Test`] —
/// it is unreachable by construction (a test token's `kid` is absent from the
/// prod JWKS and its `iss` won't match). Carry this into trust decisions so a
/// test identity can never be mistaken for a production one.
pub use capability::Environment;

/// Classify a verified issuer onto the platform-canonical
/// [`capability::Environment`]. A test issuer ends in a single non-empty
/// `/test/{tenant}` segment, and that tenant is CARRIED — everything else is
/// production.
///
/// This is a free function rather than an inherent method because
/// [`Environment`] is owned by the `capability` crate (one definition
/// platform-wide); the orphan rule puts inherent impls out of reach here, and
/// the issuer-URL rule is chirp-specific knowledge that belongs in this crate
/// rather than in the generic capability primitive.
///
/// Previously this returned a tenant-less `Test`, discarding the tenant it had
/// already parsed. Relying parties that needed it (Pigeon, chirp-auth) each
/// re-derived it with a hand-copied duplicate of this exact rule — the kind of
/// mirrored parsing that drifts. The tenant now survives the boundary.
pub fn environment_from_issuer(issuer: &str) -> Environment {
    match issuer.trim_end_matches('/').rsplit_once("/test/") {
        Some((_, tenant)) if !tenant.is_empty() && !tenant.contains('/') => Environment::Test {
            tenant: tenant.to_owned(),
        },
        _ => Environment::Prod,
    }
}

/// A verified token: its [`Environment`] (provenance — which keyset verified it)
/// alongside the [`ChirpVerifiedIdentity`] (the `Human`/`Machine` principal).
///
/// Both axes are surfaced in the type so a relying party cannot honor a test
/// identity as production, or a machine where it expects a human, by forgetting
/// to inspect a marker — the distinction is structural, not a runtime check the
/// caller might skip.
#[derive(Clone, Debug)]
pub struct ChirpVerifiedToken {
    pub environment: Environment,
    pub identity: ChirpVerifiedIdentity,
}

/// The canonical, platform-wide verified-identity carrier.
///
/// Every platform service calls [`verify_chirp_id_token`] and gets back a
/// [`ChirpVerifiedToken`] whose two axes — the [`ChirpVerifiedIdentity`]
/// principal and the [`Environment`] provenance — are *separate*. Historically
/// each consumer then collapsed both axes into its own bare-`String`
/// `AuthenticatedUser { uid }` at the service boundary, discarding the class
/// (`Human`/`Machine`/`Test`), the `owner_sub`/`client_id` of a machine caller,
/// and the keyset provenance. That made an illegal trust decision (honoring a
/// test principal as production, or a machine where a human is expected)
/// expressible — the type no longer carried what would have made it a compile
/// error.
///
/// `VerifiedIdentity` is that one carrier: it folds the principal class and the
/// environment provenance into a single value a service can store and pass
/// around *without* losing either axis. It is deliberately named
/// `VerifiedIdentity` (not `ChirpVerifiedIdentity`) to signal it is the
/// platform-canonical shape services carry, distinct from the verifier's raw
/// principal enum.
///
/// Build one from a verified token via [`VerifiedIdentity::from_token`]. Read
/// the common `sub` with [`sub`](Self::sub) and the provenance with
/// [`environment`](Self::environment); ask the class with
/// [`is_machine`](Self::is_machine) / [`is_test`](Self::is_test), or `match`
/// for the full shape (a missed class stays a compile error).
#[derive(Clone, Debug)]
pub enum VerifiedIdentity {
    /// A human sign-in (Authorization Code + PKCE). Carries only the pairwise
    /// `sub` (no email/name, by the no-email privacy design) and the keyset
    /// provenance.
    Human {
        sub: String,
        environment: Environment,
    },
    /// A confidential-client machine caller (`client_credentials`). Carries the
    /// agent `sub`, the responsible human `owner_sub`, the minting `client_id`,
    /// and the keyset provenance.
    Machine {
        sub: String,
        owner_sub: String,
        client_id: String,
        environment: Environment,
    },
    /// A test-keyset principal (verified against a `…/test/{tenant}` issuer).
    /// Structurally distinct from `Human` so a production surface cannot honor
    /// it by forgetting to inspect the environment axis.
    Test {
        sub: String,
        environment: Environment,
    },
}

impl VerifiedIdentity {
    /// Fold a verified token's principal + provenance into the canonical
    /// carrier. Total over [`ChirpVerifiedIdentity`]'s variants.
    pub fn from_token(token: ChirpVerifiedToken) -> Self {
        let environment = token.environment;
        match token.identity {
            ChirpVerifiedIdentity::Human { sub } => Self::Human { sub, environment },
            ChirpVerifiedIdentity::Machine {
                sub,
                owner_sub,
                client_id,
            } => Self::Machine {
                sub,
                owner_sub,
                client_id,
                environment,
            },
            ChirpVerifiedIdentity::Test { sub } => Self::Test { sub, environment },
        }
    }

    /// The token's `sub` claim regardless of variant.
    pub fn sub(&self) -> &str {
        match self {
            Self::Human { sub, .. } | Self::Machine { sub, .. } | Self::Test { sub, .. } => sub,
        }
    }

    /// The keyset provenance ([`Environment`]) this identity was verified under.
    ///
    /// Returns a reference: [`Environment`] carries the test tenant and so is
    /// no longer `Copy`. Callers that need an owned value can `.clone()`.
    pub fn environment(&self) -> &Environment {
        match self {
            Self::Human { environment, .. }
            | Self::Machine { environment, .. }
            | Self::Test { environment, .. } => environment,
        }
    }

    /// `true` for a [`Machine`](Self::Machine) caller.
    pub fn is_machine(&self) -> bool {
        matches!(self, Self::Machine { .. })
    }

    /// `true` for a [`Test`](Self::Test) principal.
    pub fn is_test(&self) -> bool {
        matches!(self, Self::Test { .. })
    }
}

/// Per-call knobs for [`verify_chirp_id_token`].
///
/// Defaults to human-only acceptance — machine tokens are rejected with
/// [`ChirpAuthError::MachineNotAllowed`] unless `accept_machine` is set.
/// Services that want to participate in the machine-identity protocol opt in
/// here; everyone else stays human-only with no code change.
///
/// `#[non_exhaustive]`: construct via `VerifyOptions { .. ..Default::default() }`
/// so future option additions stay non-breaking.
#[derive(Default, Clone, Debug)]
#[non_exhaustive]
pub struct VerifyOptions {
    /// When `true`, accept machine tokens (`act == "machine"`) in addition to
    /// human tokens. Default `false`.
    pub accept_machine: bool,
    /// The set of machine `client_id`s this relying party will act on.
    ///
    /// A machine token's `client_id` is its own minting client (its `aud`); it
    /// is an audience-free bearer assertion presentable to any service, so the
    /// human-path audience allowlist does NOT gate it. Each consuming service
    /// instead used to hand-roll a membership check on the returned `client_id`
    /// (Drive/Granite/Social Graph all duplicated the same code). That rule now
    /// lives here, in one audited place.
    ///
    /// When `accept_machine` is `true`, the verified machine `client_id` MUST be
    /// a member of this set or the token is rejected with
    /// [`ChirpAuthError::MachineAudienceNotAccepted`]. The set **fails closed**:
    /// an empty set accepts no machine token (so a service that opts into
    /// machine acceptance without naming any client gets nothing through, rather
    /// than silently trusting every agent). Ignored when `accept_machine` is
    /// `false`, or when `accept_any_machine_client` is `true`.
    pub accepted_machine_audiences: BTreeSet<String>,
    /// Escape hatch for **grant-scoped** relying parties: when `true` (and
    /// `accept_machine` is `true`), the [`accepted_machine_audiences`] membership
    /// check is bypassed and a machine token from ANY `client_id` verifies.
    ///
    /// This is the opposite of the fail-closed default and is ONLY sound for a
    /// service whose authorization does not depend on *which* agent authenticated
    /// — i.e. one that gates every privileged/data operation on a separate,
    /// explicit grant (the grant ledger, not a static client_id set, is its
    /// allowlist). Drive is the canonical case: it accepts any authenticated
    /// agent for grant-scoped access, then requires an owner grant per operation.
    /// A service that trusts the verified `client_id` for anything MUST leave
    /// this `false` and enumerate [`accepted_machine_audiences`] instead.
    ///
    /// [`accepted_machine_audiences`]: VerifyOptions::accepted_machine_audiences
    pub accept_any_machine_client: bool,
}

impl VerifyOptions {
    /// Build options that accept machine tokens minted by any of `client_ids`.
    ///
    /// Convenience over the struct literal for the common "accept machine, gated
    /// to these clients" case. Equivalent to
    /// `VerifyOptions { accept_machine: true, accepted_machine_audiences: …, ..Default::default() }`.
    /// An empty iterator yields an opt-in that accepts no machine token
    /// (fail-closed); see [`accepted_machine_audiences`].
    ///
    /// [`accepted_machine_audiences`]: VerifyOptions::accepted_machine_audiences
    pub fn accept_machine_clients(client_ids: impl IntoIterator<Item = String>) -> Self {
        VerifyOptions {
            accept_machine: true,
            accepted_machine_audiences: client_ids
                .into_iter()
                .map(|id| id.trim().to_owned())
                .collect(),
            ..Default::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChirpAuthError {
    #[error("malformed jwt")]
    MalformedToken,
    #[error("unsupported alg")]
    UnsupportedAlgorithm,
    #[error("jwks fetch failed: {0}")]
    JwksFetch(String),
    #[error("no matching key")]
    NoMatchingKey,
    #[error("signature verification failed")]
    SignatureInvalid,
    #[error("issuer/audience claim mismatch")]
    ClaimMismatch,
    #[error("token expired")]
    Expired,
    #[error("invalid subject")]
    InvalidSubject,
    #[error("machine token not accepted by this endpoint")]
    MachineNotAllowed,
    /// The token verified as a machine identity, but its `client_id` is not in
    /// [`VerifyOptions::accepted_machine_audiences`]. Distinct from
    /// [`MachineNotAllowed`] (which means this RP accepts no machine tokens at
    /// all): here machine tokens ARE accepted, just not from this client.
    #[error("machine token client_id not in accepted-machine-audiences set")]
    MachineAudienceNotAccepted,
    #[error("machine token missing owner_sub")]
    MalformedMachineToken,
    /// The token's provenance (production vs. test) disagrees with the
    /// configured issuer's [`Environment`]. A production RP got a `test` token,
    /// or a test RP got a non-`test` token.
    #[error("token environment does not match configured issuer")]
    EnvironmentMismatch,
    /// No `Authorization: Bearer …` header, or an empty/whitespace token.
    #[error("missing or empty bearer token")]
    MissingBearer,
    /// The JWT `typ` header is PRESENT but does not match the token class this
    /// verifier expects (e.g. a key-binding cert presented as an ID token).
    /// Header-level class isolation, distinct from any claim-shape rejection. A
    /// FULLY ABSENT `typ` never reaches this check: `typ` is a required header
    /// field, so a header without it fails to deserialize and is rejected as
    /// [`MalformedToken`](Self::MalformedToken). See [`ID_TOKEN_TYP`] /
    /// [`KEYBIND_TYP`].
    #[error("typ header does not match the expected token class")]
    InvalidTokenType,
}

/// JWT `typ` for the ChirpAuth ID-token class (human / machine / test). The
/// issuer stamps this on every ID token; this verifier requires an exact match.
pub const ID_TOKEN_TYP: &str = "chirp-id+jwt";
/// JWT `typ` for ChirpAuth key-binding certificates. Pass to [`verify_rs256_jws`]
/// when verifying a cert so a cert can't be honored as an ID token (or vice
/// versa).
pub const KEYBIND_TYP: &str = "chirp-keybind+jwt";
/// JWT `typ` for ChirpAuth resource-scoped access tokens (RFC 8707): `aud` is a
/// RESOURCE (e.g. `drive`), not the requesting client, and the token carries a
/// space-separated `scope` claim. A resource server verifies one of these with
/// [`verify_chirp_resource_access_token`]. Distinct from [`ID_TOKEN_TYP`] so an
/// RP ID token can never be honored as resource authority (or vice versa).
pub const ACCESS_TOKEN_TYP: &str = "chirp-access+jwt";

#[derive(Debug, Deserialize)]
struct JwtHeader {
    alg: String,
    kid: String,
    /// Token-class marker. Required (no `#[serde(default)]`): a header without a
    /// `typ` (pre-hardening) fails to deserialize and is rejected as
    /// [`ChirpAuthError::MalformedToken`] before any class check — there is no
    /// legacy/absent acceptance. Verifiers assert a PRESENT `typ` equals their
    /// expected class string (a mismatch → [`ChirpAuthError::InvalidTokenType`]).
    typ: String,
}

#[derive(Debug, Deserialize)]
struct ChirpClaims {
    iss: String,
    sub: String,
    aud: AudienceClaim,
    exp: i64,
    #[serde(default)]
    act: Option<String>,
    #[serde(default)]
    owner_sub: Option<String>,
    #[serde(default)]
    test: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AudienceClaim {
    One(String),
    Many(Vec<String>),
}

impl AudienceClaim {
    /// True iff at least one audience value is in the allowlist.
    fn matches_any(&self, allowed: &BTreeSet<String>) -> bool {
        match self {
            Self::One(audience) => allowed.contains(audience),
            Self::Many(audiences) => audiences.iter().any(|a| allowed.contains(a)),
        }
    }

    fn single(&self) -> Option<&str> {
        match self {
            Self::One(audience) => Some(audience.as_str()),
            Self::Many(audiences) if audiences.len() == 1 => Some(audiences[0].as_str()),
            Self::Many(_) => None,
        }
    }
}

/// A parsed JWKS document — the set of RSA signing keys ChirpAuth publishes.
/// Returned by [`fetch_jwks`] for callers that need direct key access.
#[derive(Debug, Clone, Deserialize)]
pub struct Jwks {
    pub keys: Vec<Jwk>,
}

/// A single JWK (RSA public key) from a [`Jwks`] document.
#[derive(Debug, Clone, Deserialize)]
pub struct Jwk {
    pub kty: String,
    pub kid: String,
    pub alg: Option<String>,
    pub n: String,
    pub e: String,
}

struct JwtParts<'a> {
    header: &'a str,
    claims: &'a str,
    signature: &'a str,
    signing_input: &'a str,
}

fn jwt_parts(token: &str) -> Result<JwtParts<'_>, ChirpAuthError> {
    let mut segments = token.split('.');
    let header = segments.next().ok_or(ChirpAuthError::MalformedToken)?;
    let claims = segments.next().ok_or(ChirpAuthError::MalformedToken)?;
    let signature = segments.next().ok_or(ChirpAuthError::MalformedToken)?;
    if segments.next().is_some()
        || header.is_empty()
        || claims.is_empty()
        || signature.is_empty()
    {
        return Err(ChirpAuthError::MalformedToken);
    }
    let signing_input_len = header.len() + 1 + claims.len();
    Ok(JwtParts {
        header,
        claims,
        signature,
        signing_input: &token[..signing_input_len],
    })
}

fn verify_rsa_signature(
    key: &Jwk,
    signing_input: &[u8],
    signature: &[u8],
) -> Result<(), ChirpAuthError> {
    let n = URL_SAFE_NO_PAD
        .decode(&key.n)
        .map_err(|_| ChirpAuthError::SignatureInvalid)?;
    let e = URL_SAFE_NO_PAD
        .decode(&key.e)
        .map_err(|_| ChirpAuthError::SignatureInvalid)?;
    RsaPublicKeyComponents { n: &n, e: &e }
        .verify(&RSA_PKCS1_2048_8192_SHA256, signing_input, signature)
        .map_err(|_| ChirpAuthError::SignatureInvalid)
}

/// Extract a bearer token from an `Authorization` header.
///
/// Returns the token with the `Bearer ` prefix stripped and surrounding
/// whitespace trimmed, or `None` if the header is absent, not a `Bearer`
/// scheme, or the token is empty after trimming. The scheme match is
/// ASCII-case-insensitive per RFC 7235.
pub fn bearer_token(headers: &http::HeaderMap) -> Option<&str> {
    let value = headers.get(http::header::AUTHORIZATION)?.to_str().ok()?;
    let rest = value.strip_prefix("Bearer ").or_else(|| {
        // Case-insensitive scheme match without allocating: split on the first
        // space and compare the scheme.
        let (scheme, rest) = value.split_once(' ')?;
        scheme.eq_ignore_ascii_case("bearer").then_some(rest)
    })?;
    let token = rest.trim();
    (!token.is_empty()).then_some(token)
}

/// Fetch and parse the JWKS document at `config.jwks_uri`.
///
/// Exposed so key-binding-certificate verification (and any other downstream
/// RS256 use) can reuse one fetch+parse path instead of re-implementing it.
pub async fn fetch_jwks(
    client: &reqwest::Client,
    config: &ChirpAuthConfig,
) -> Result<Jwks, ChirpAuthError> {
    client
        .get(&config.jwks_uri)
        .send()
        .await
        .map_err(|error| ChirpAuthError::JwksFetch(error.to_string()))?
        .error_for_status()
        .map_err(|error| ChirpAuthError::JwksFetch(error.to_string()))?
        .json::<Jwks>()
        .await
        .map_err(|error| ChirpAuthError::JwksFetch(error.to_string()))
}

/// A verified RS256 JWS payload — the decoded claims after signature, `iss`,
/// `exp`, and (optionally) `aud` checks pass. Returned by
/// [`verify_rs256_jws`].
#[derive(Clone, Debug)]
pub struct Claims {
    pub iss: String,
    pub sub: String,
    pub aud: Vec<String>,
    pub exp: i64,
    /// The raw JSON object of all claims, for callers that need claims beyond
    /// the standard set (e.g. a key-binding cert's embedded public key).
    pub raw: serde_json::Map<String, serde_json::Value>,
}

/// Verify an RS256 JWS against `config`'s JWKS and standard claims.
///
/// This is the low-level verifier [`verify_chirp_id_token`] is built on, exposed
/// so downstream verification of *other* ChirpAuth-signed artifacts (notably
/// key-binding certificates) reuses one audited RS256 path rather than
/// re-implementing JWKS fetch + signature checks.
///
/// Checks: RS256 alg pin (header and JWK), the `typ` header equals
/// `expected_typ`, kid lookup in the JWKS, signature, exact `iss ==
/// config.issuer`, and `exp` in the future. When `validate_aud` is `true`, also
/// requires at least one `aud` value in [`ChirpAuthConfig::accepted_audiences`].
/// Does **not** apply machine/test policy — that lives in
/// [`verify_chirp_id_token`].
///
/// `expected_typ` is the token-class header this caller accepts (e.g.
/// [`KEYBIND_TYP`] for a key-binding cert). A token whose `typ` is PRESENT but
/// differs is rejected with [`ChirpAuthError::InvalidTokenType`]; a token with NO
/// `typ` header fails header deserialization and is rejected as
/// [`ChirpAuthError::MalformedToken`]. Either way, one ChirpAuth-signed artifact
/// class can never be honored as another.
pub async fn verify_rs256_jws(
    client: &reqwest::Client,
    config: &ChirpAuthConfig,
    token: &str,
    validate_aud: bool,
    expected_typ: &str,
) -> Result<Claims, ChirpAuthError> {
    let parts = jwt_parts(token)?;
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts.header)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let claims_bytes = URL_SAFE_NO_PAD
        .decode(parts.claims)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let signature = URL_SAFE_NO_PAD
        .decode(parts.signature)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    if header.alg != "RS256" {
        return Err(ChirpAuthError::UnsupportedAlgorithm);
    }
    if header.typ != expected_typ {
        return Err(ChirpAuthError::InvalidTokenType);
    }

    let jwks = fetch_jwks(client, config).await?;
    let key = jwks
        .keys
        .iter()
        .find(|key| key.kid == header.kid && key.kty == "RSA")
        .ok_or(ChirpAuthError::NoMatchingKey)?;
    if key.alg.as_deref().is_some_and(|alg| alg != "RS256") {
        return Err(ChirpAuthError::UnsupportedAlgorithm);
    }
    verify_rsa_signature(key, parts.signing_input.as_bytes(), &signature)?;

    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&claims_bytes).map_err(|_| ChirpAuthError::MalformedToken)?;
    let claims: ChirpClaims =
        serde_json::from_slice(&claims_bytes).map_err(|_| ChirpAuthError::MalformedToken)?;

    if claims.iss != config.issuer {
        return Err(ChirpAuthError::ClaimMismatch);
    }
    if validate_aud && !claims.aud.matches_any(&config.accepted_audiences) {
        return Err(ChirpAuthError::ClaimMismatch);
    }
    if claims.exp <= OffsetDateTime::now_utc().unix_timestamp() {
        return Err(ChirpAuthError::Expired);
    }

    let aud = match &claims.aud {
        AudienceClaim::One(a) => vec![a.clone()],
        AudienceClaim::Many(many) => many.clone(),
    };
    Ok(Claims {
        iss: claims.iss,
        sub: claims.sub,
        aud,
        exp: claims.exp,
        raw,
    })
}

/// Extract the bearer token from `headers` and verify it in one call.
///
/// Convenience over [`bearer_token`] + [`verify_chirp_id_token`] so relying
/// parties stop hand-rolling that two-step in middleware. Returns the
/// [`ChirpVerifiedIdentity`] directly; callers needing the [`Environment`]
/// should use the lower-level pair. [`ChirpAuthError::MissingBearer`] when no
/// usable bearer token is present.
pub async fn verify_from_headers(
    client: &reqwest::Client,
    headers: &http::HeaderMap,
    config: &ChirpAuthConfig,
    options: VerifyOptions,
) -> Result<ChirpVerifiedIdentity, ChirpAuthError> {
    let token = bearer_token(headers).ok_or(ChirpAuthError::MissingBearer)?;
    verify_chirp_id_token(client, config, token, options)
        .await
        .map(|verified| verified.identity)
}

/// A verified resource-scoped access token (RFC 8707): the human `sub` acting on
/// a RESOURCE, the `resource` the token is audience'd to (the matched audience),
/// and the granted `scopes`, alongside the [`Environment`] that verified it.
///
/// Unlike [`ChirpVerifiedIdentity`], this is NOT tied to the requesting client:
/// a resource server validates `resource` (`aud`) + `scopes` and does not care
/// which client obtained the token — that is the whole point of the
/// resource-audience design (see chirp-auth
/// `protocols/docs/adr-resource-scoped-access-tokens.md`). The requesting client
/// rides as `azp` in [`Claims::raw`] for audit if a caller wants it.
#[derive(Clone, Debug)]
pub struct ChirpVerifiedResourceAccess {
    pub environment: Environment,
    pub sub: String,
    pub resource: String,
    pub scopes: BTreeSet<String>,
}

impl ChirpVerifiedResourceAccess {
    /// Whether the token grants `scope` (e.g. `"drive.deposit"`).
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.contains(scope)
    }
}

/// Verify a ChirpAuth resource-scoped access token (`chirp-access+jwt`) end-to-end.
///
/// The audience allowlist ([`ChirpAuthConfig::accepted_audiences`]) is the set of
/// RESOURCES this server serves (e.g. `{"drive"}`); the token is accepted only if
/// its `aud` names one of them. On success the matched resource, the human `sub`,
/// and the granted `scopes` are returned — the caller then checks it
/// [`has_scope`](ChirpVerifiedResourceAccess::has_scope) for the action it gates.
/// A resource token with no `scope` grants nothing and is rejected.
///
/// This is a thin, audited wrapper over [`verify_rs256_jws`] with
/// `expected_typ = `[`ACCESS_TOKEN_TYP`] and `validate_aud = true`, so an ID
/// token (or any other class) can never be honored here.
pub async fn verify_chirp_resource_access_token(
    client: &reqwest::Client,
    config: &ChirpAuthConfig,
    token: &str,
) -> Result<ChirpVerifiedResourceAccess, ChirpAuthError> {
    let claims = verify_rs256_jws(client, config, token, true, ACCESS_TOKEN_TYP).await?;
    let scopes: BTreeSet<String> = claims
        .raw
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if scopes.is_empty() {
        // A resource token that grants nothing is malformed for our purposes.
        return Err(ChirpAuthError::ClaimMismatch);
    }
    // The resource is the token's audience that this server accepts. `aud` is
    // single-valued for access tokens; pick the value in our allowlist.
    let resource = claims
        .aud
        .iter()
        .find(|a| config.accepted_audiences.contains(*a))
        .cloned()
        .ok_or(ChirpAuthError::ClaimMismatch)?;
    Ok(ChirpVerifiedResourceAccess {
        environment: environment_from_issuer(&claims.iss),
        sub: claims.sub,
        resource,
        scopes,
    })
}

/// Verify a ChirpAuth-issued RS256 ID token end-to-end.
///
/// Fetches `config.jwks_uri` each call — callers that want caching should layer
/// it on top (ChirpAuth JWKS rotation is on the order of hours/days; per-request
/// fetch is fine for current call volumes but worth revisiting if hot).
///
/// Returns a [`ChirpVerifiedIdentity`] — `Machine` when `act == "machine"`,
/// else `Test` under a test issuer or `Human` under a production issuer (the
/// identity class follows the verified keyset's environment). Callers that don't
/// accept machine identities leave `options.accept_machine = false` (the
/// default) and never see a `Machine` arm in practice — the verifier rejects
/// with [`ChirpAuthError::MachineNotAllowed`] first. A production RP never sees
/// `Test` (environment enforcement rejects a `test` token first); a test RP
/// never sees `Human`.
///
/// **Environment is enforced** (0.7): the expected [`Environment`] is derived
/// from `config`'s issuer, and a token whose provenance disagrees is rejected
/// with [`ChirpAuthError::EnvironmentMismatch`]. A production RP rejects a
/// `test == true` token; a test RP rejects a non-test token. This replaces the
/// former per-call `accept_test` flag.
///
/// Errors are mapped granularly so the caller can decide which to surface as
/// "401 Unauthorized" vs "500 Internal" — most consumers collapse everything
/// to 401, which is correct.
pub async fn verify_chirp_id_token(
    client: &reqwest::Client,
    config: &ChirpAuthConfig,
    token: &str,
    options: VerifyOptions,
) -> Result<ChirpVerifiedToken, ChirpAuthError> {
    let parts = jwt_parts(token)?;
    let header_bytes = URL_SAFE_NO_PAD
        .decode(parts.header)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let claims_bytes = URL_SAFE_NO_PAD
        .decode(parts.claims)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let signature = URL_SAFE_NO_PAD
        .decode(parts.signature)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    if header.alg != "RS256" {
        return Err(ChirpAuthError::UnsupportedAlgorithm);
    }
    // Header-level class isolation: an ID token MUST carry `typ: chirp-id+jwt`.
    // A key-binding cert (`chirp-keybind+jwt`), or any other PRESENT-but-wrong
    // `typ`, is rejected here as `InvalidTokenType` before any claim inspection.
    // A pre-hardening token with NO `typ` never reaches this line: `typ` is a
    // required header field, so its absence already failed header deserialization
    // above as `MalformedToken`. Human / machine / test ID tokens all share this
    // one typ; the human-vs-machine split is the `act` claim, the prod-vs-test
    // split is the issuer-derived environment.
    if header.typ != ID_TOKEN_TYP {
        return Err(ChirpAuthError::InvalidTokenType);
    }

    let jwks = fetch_jwks(client, config).await?;
    let key = jwks
        .keys
        .iter()
        .find(|key| key.kid == header.kid && key.kty == "RSA")
        .ok_or(ChirpAuthError::NoMatchingKey)?;
    if key.alg.as_deref().is_some_and(|alg| alg != "RS256") {
        return Err(ChirpAuthError::UnsupportedAlgorithm);
    }
    verify_rsa_signature(key, parts.signing_input.as_bytes(), &signature)?;

    let claims: ChirpClaims = serde_json::from_slice(&claims_bytes)
        .map_err(|_| ChirpAuthError::MalformedToken)?;
    if claims.iss != config.issuer {
        return Err(ChirpAuthError::ClaimMismatch);
    }
    // Audience rule, by token shape (NOT by "human vs machine"):
    //   - A human id_token (authorization_code) is minted FOR one relying party,
    //     so its `aud` must be in this RP's accepted set.
    //   - A machine token (client_credentials) is an audience-free bearer
    //     assertion of an agent identity: its `aud` is just the minting client's
    //     own id, presentable to any service. Audience is NOT enforced here; the
    //     verifying service authorizes the agent via its own capability ledger.
    let is_machine = matches!(claims.act.as_deref(), Some("machine"));
    if !is_machine && !claims.aud.matches_any(&config.accepted_audiences) {
        return Err(ChirpAuthError::ClaimMismatch);
    }
    if claims.exp <= OffsetDateTime::now_utc().unix_timestamp() {
        return Err(ChirpAuthError::Expired);
    }

    // Provenance enforcement: the environment is fixed by the configured issuer
    // (hence which keyset verified this token — `claims.iss` was just confirmed
    // to equal `config.issuer`). It is the authoritative axis; the `test` claim
    // is a defense-in-depth cross-check whose form differs by token shape:
    //
    //   - Non-machine (human) tokens carry `test: true` iff they are test tokens,
    //     so the claim must AGREE with the issuer-derived environment:
    //       production issuer ⇒ must NOT be `test: true`;
    //       test issuer       ⇒ MUST be `test: true`.
    //   - Machine tokens NEVER carry `test: true` — `act: "machine"` + `test: true`
    //     is unrepresentable at mint (INV-09); a test-keyset machine token is a
    //     `Machine` principal whose test-ness is purely its issuer/keyset. So for
    //     machines the environment comes from provenance alone, and a machine
    //     token that smuggles `test: true` is rejected as malformed.
    //
    // Either disagreement fails closed, keeping the environment axis a hard
    // boundary rather than a per-call opt-in a caller might forget.
    let environment = config.environment();
    let token_is_test = claims.test == Some(true);
    if is_machine {
        if token_is_test {
            return Err(ChirpAuthError::MalformedMachineToken);
        }
    } else {
        // Compare only the test-ness axis. The token's `test` claim is a bare
        // bool and names no tenant; the TENANT comes solely from the issuer,
        // which is authoritative because that issuer's per-tenant JWKS is what
        // verified this signature. Deriving the tenant from the keyset that
        // actually validated the token — rather than from anything the token
        // asserts about itself — is what keeps it unforgeable.
        let config_is_test = matches!(environment, Environment::Test { .. });
        if token_is_test != config_is_test {
            return Err(ChirpAuthError::EnvironmentMismatch);
        }
    }

    let sub = claims.sub.trim().to_owned();
    if sub.is_empty() || sub.chars().count() > 128 || sub.chars().any(char::is_control) {
        return Err(ChirpAuthError::InvalidSubject);
    }

    if is_machine {
        if !options.accept_machine {
            return Err(ChirpAuthError::MachineNotAllowed);
        }
        let owner_sub = claims
            .owner_sub
            .as_ref()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .ok_or(ChirpAuthError::MalformedMachineToken)?;
        // Machine tokens MUST address a single client (the one that minted them).
        let client_id = claims
            .aud
            .single()
            .map(str::to_owned)
            .ok_or(ChirpAuthError::MalformedMachineToken)?;
        // Accepted-machine-audiences gate. The human-path audience allowlist
        // does NOT apply to machine tokens (their `aud` is the minting client,
        // not this RP), so the only place a service can scope WHICH agents may
        // act on it is the verified `client_id`. Consumers used to re-implement
        // this membership check each (Drive/Granite/SG); it now lives here,
        // once. Fails closed: an empty accepted set lets no machine token in.
        // ...UNLESS the RP opts into grant-scoped acceptance of any agent, where
        // authorization is its own per-operation grant, not the client_id.
        if !options.accept_any_machine_client
            && !options.accepted_machine_audiences.contains(&client_id)
        {
            return Err(ChirpAuthError::MachineAudienceNotAccepted);
        }
        return Ok(ChirpVerifiedToken {
            environment,
            identity: ChirpVerifiedIdentity::Machine {
                sub,
                owner_sub,
                client_id,
            },
        });
    }

    // Non-machine ID token. The principal class follows the verified
    // environment (provenance): a test-keyset token is a `Test` principal, a
    // production-keyset token is a `Human`. Environment enforcement above has
    // already rejected any token whose `test` claim disagrees with the
    // configured issuer, so this match is exhaustive over the reachable cases:
    // a production RP only ever reaches `Human`, a test RP only `Test`.
    let identity = match environment {
        Environment::Prod => ChirpVerifiedIdentity::Human { sub },
        Environment::Test { .. } => ChirpVerifiedIdentity::Test { sub },
    };
    Ok(ChirpVerifiedToken {
        environment,
        identity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_with_prefix_reads_prefixed_vars() {
        // Use a unique prefix to avoid clobbering anything else.
        let prefix = "CHIRP_AUTH_CLIENT_TEST_PREFIX_A";
        // SAFETY: tests in this module are single-threaded by default per binary.
        unsafe {
            std::env::set_var(format!("{prefix}_ISSUER"), "https://example.test/");
            std::env::set_var(format!("{prefix}_AUDIENCE"), "drive");
        }
        let config = ChirpAuthConfig::from_env(prefix).expect("config");
        assert_eq!(config.issuer(), "https://example.test");
        assert!(config.accepted_audiences().contains("drive"));
        assert_eq!(config.jwks_uri(), "https://example.test/jwks.json");
        unsafe {
            std::env::remove_var(format!("{prefix}_ISSUER"));
            std::env::remove_var(format!("{prefix}_AUDIENCE"));
        }
    }

    #[test]
    fn from_env_returns_none_when_missing() {
        let prefix = "CHIRP_AUTH_CLIENT_TEST_PREFIX_MISSING";
        // ensure unset
        unsafe {
            std::env::remove_var(format!("{prefix}_ISSUER"));
            std::env::remove_var(format!("{prefix}_AUDIENCE"));
        }
        assert!(ChirpAuthConfig::from_env(prefix).is_none());
    }

    #[test]
    fn from_env_returns_none_when_empty() {
        let prefix = "CHIRP_AUTH_CLIENT_TEST_PREFIX_EMPTY";
        unsafe {
            std::env::set_var(format!("{prefix}_ISSUER"), "   ");
            std::env::set_var(format!("{prefix}_AUDIENCE"), "drive");
        }
        assert!(ChirpAuthConfig::from_env(prefix).is_none());
        unsafe {
            std::env::remove_var(format!("{prefix}_ISSUER"));
            std::env::remove_var(format!("{prefix}_AUDIENCE"));
        }
    }

    #[test]
    fn new_derives_jwks_uri_and_normalizes_issuer() {
        let config = ChirpAuthConfig::new("https://signin.chirpauth.com/", "drive");
        assert_eq!(config.issuer(), "https://signin.chirpauth.com");
        assert_eq!(config.jwks_uri(), "https://signin.chirpauth.com/jwks.json");
        assert_eq!(config.environment(), Environment::Prod);
    }

    #[test]
    fn default_issuer_is_production_environment() {
        let config = ChirpAuthConfig::new(DEFAULT_ISSUER, "drive");
        assert_eq!(config.environment(), Environment::Prod);
        assert_eq!(config.issuer(), "https://signin.chirpauth.com");
    }

    #[test]
    fn with_audiences_builds_allowlist() {
        let config = ChirpAuthConfig::with_audiences(
            DEFAULT_ISSUER,
            ["a".to_owned(), "b".to_owned(), "c".to_owned()],
        );
        assert!(config.accepted_audiences().contains("a"));
        assert!(config.accepted_audiences().contains("b"));
        assert!(config.accepted_audiences().contains("c"));
        assert!(!config.accepted_audiences().contains("d"));
    }

    #[test]
    fn add_audience_extends_allowlist() {
        let config = ChirpAuthConfig::new(DEFAULT_ISSUER, "a").add_audience("b");
        assert!(config.accepted_audiences().contains("a"));
        assert!(config.accepted_audiences().contains("b"));
    }

    #[tokio::test]
    async fn rejects_malformed_token() {
        let config = ChirpAuthConfig::new("https://example.test", "drive")
            .with_jwks_uri("https://example.test/jwks.json");
        let client = reqwest::Client::new();
        let err = verify_chirp_id_token(&client, &config, "not.a.jwt.too.many", VerifyOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ChirpAuthError::MalformedToken));
    }

    #[tokio::test]
    async fn rejects_two_segment_token() {
        let config = ChirpAuthConfig::new("https://example.test", "drive")
            .with_jwks_uri("https://example.test/jwks.json");
        let client = reqwest::Client::new();
        let err = verify_chirp_id_token(&client, &config, "abc.def", VerifyOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ChirpAuthError::MalformedToken));
    }

    #[test]
    fn chirp_claims_parses_test_flag_when_present_and_absent() {
        // Absent → None, which the verifier treats as production provenance.
        let absent: ChirpClaims = serde_json::from_str(
            r#"{"iss":"x","sub":"sub_a","aud":"x","exp":1}"#,
        )
        .expect("parse");
        assert_eq!(absent.test, None);

        // Present and true → the value the verifier gates environment on.
        let present: ChirpClaims = serde_json::from_str(
            r#"{"iss":"x","sub":"sub_a","aud":"x","exp":1,"test":true}"#,
        )
        .expect("parse");
        assert_eq!(present.test, Some(true));
    }

    #[test]
    fn identity_sub_returns_underlying_sub() {
        let human = ChirpVerifiedIdentity::Human { sub: "sub_abc".into() };
        assert_eq!(human.sub(), "sub_abc");
        let machine = ChirpVerifiedIdentity::Machine {
            sub: "agent_xyz".into(),
            owner_sub: "sub_owner".into(),
            client_id: "cs_live_1".into(),
        };
        assert_eq!(machine.sub(), "agent_xyz");
        let test = ChirpVerifiedIdentity::Test { sub: "sub_test_principal".into() };
        assert_eq!(test.sub(), "sub_test_principal");
    }

    #[test]
    fn verified_identity_folds_token_axes() {
        // Human: carries sub + provenance, classifies as neither machine nor test.
        let human = VerifiedIdentity::from_token(ChirpVerifiedToken {
            environment: Environment::Prod,
            identity: ChirpVerifiedIdentity::Human { sub: "sub_h".into() },
        });
        assert_eq!(human.sub(), "sub_h");
        assert_eq!(human.environment(), &Environment::Prod);
        assert!(!human.is_machine());
        assert!(!human.is_test());

        // Machine: owner_sub + client_id survive the fold (previously discarded).
        let machine = VerifiedIdentity::from_token(ChirpVerifiedToken {
            environment: Environment::Prod,
            identity: ChirpVerifiedIdentity::Machine {
                sub: "agent".into(),
                owner_sub: "sub_owner".into(),
                client_id: "cs_live_1".into(),
            },
        });
        assert_eq!(machine.sub(), "agent");
        assert!(machine.is_machine());
        match machine {
            VerifiedIdentity::Machine {
                owner_sub,
                client_id,
                ..
            } => {
                assert_eq!(owner_sub, "sub_owner");
                assert_eq!(client_id, "cs_live_1");
            }
            _ => panic!("expected machine"),
        }

        // Test: provenance is Test and the class is distinguishable.
        let test = VerifiedIdentity::from_token(ChirpVerifiedToken {
            environment: Environment::Test { tenant: "acme".into() },
            identity: ChirpVerifiedIdentity::Test { sub: "sub_t".into() },
        });
        assert_eq!(test.sub(), "sub_t");
        assert_eq!(test.environment(), &Environment::Test { tenant: "acme".into() });
        assert!(test.is_test());
    }

    // -------------------- bearer extraction --------------------

    fn headers_with_auth(value: &str) -> http::HeaderMap {
        let mut h = http::HeaderMap::new();
        h.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_str(value).unwrap(),
        );
        h
    }

    #[test]
    fn bearer_token_strips_prefix_and_trims() {
        let h = headers_with_auth("Bearer   abc.def.ghi  ");
        assert_eq!(bearer_token(&h), Some("abc.def.ghi"));
    }

    #[test]
    fn bearer_token_case_insensitive_scheme() {
        let h = headers_with_auth("bearer tok123");
        assert_eq!(bearer_token(&h), Some("tok123"));
        let h = headers_with_auth("BEARER tok456");
        assert_eq!(bearer_token(&h), Some("tok456"));
    }

    #[test]
    fn bearer_token_rejects_empty_after_prefix() {
        let h = headers_with_auth("Bearer    ");
        assert_eq!(bearer_token(&h), None);
        let h = headers_with_auth("Bearer ");
        assert_eq!(bearer_token(&h), None);
    }

    #[test]
    fn bearer_token_rejects_other_schemes() {
        let h = headers_with_auth("Basic dXNlcjpwYXNz");
        assert_eq!(bearer_token(&h), None);
        let h = headers_with_auth("Token abc");
        assert_eq!(bearer_token(&h), None);
    }

    #[test]
    fn bearer_token_none_when_header_absent() {
        let h = http::HeaderMap::new();
        assert_eq!(bearer_token(&h), None);
    }

    #[test]
    fn bearer_token_rejects_bare_scheme_word() {
        // "Bearer" with no space/value is not a valid bearer credential.
        let h = headers_with_auth("Bearer");
        assert_eq!(bearer_token(&h), None);
    }
}

// ---------------------------------------------------------------------------
// End-to-end verify-path tests.
//
// The tests above this module exercise env-var parsing and malformed-shortcut
// paths that fail before any JWKS fetch or signature check. The tests below
// stand up an in-process JWKS server, mint real RS256 tokens against a
// generated keypair, and assert the verifier behaves correctly on the full
// path: signature, kid lookup, algorithm pin, issuer/audience claim, expiry,
// machine-token gating, environment enforcement, and a handful of adversarial
// constructions (alg=none, alg-confusion via RS512, kid not in JWKS,
// issuer-suffix attack).
//
// One RSA-2048 keypair is generated lazily and shared across tests via
// `OnceLock` — RSA keygen is the slow part (~200ms); reusing the key keeps
// the suite cheap.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod verify_path_tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use rsa::pkcs1v15::SigningKey;
    use rsa::signature::{RandomizedSigner, SignatureEncoding};
    use rsa::traits::PublicKeyParts;
    use rsa::{RsaPrivateKey, RsaPublicKey};
    use sha2::Sha256;
    use std::sync::OnceLock;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const KID: &str = "test-kid-1";
    const ISS: &str = "https://signin.test.example";
    const AUD: &str = "cs_test_audience";

    fn keypair() -> &'static RsaPrivateKey {
        static KEY: OnceLock<RsaPrivateKey> = OnceLock::new();
        KEY.get_or_init(|| {
            let mut rng = rand::thread_rng();
            RsaPrivateKey::new(&mut rng, 2048).expect("generate test RSA key")
        })
    }

    fn jwks_body_with_test_key() -> String {
        let pubkey = RsaPublicKey::from(keypair());
        let n = URL_SAFE_NO_PAD.encode(pubkey.n().to_bytes_be());
        let e = URL_SAFE_NO_PAD.encode(pubkey.e().to_bytes_be());
        format!(
            r#"{{"keys":[{{"kty":"RSA","kid":"{KID}","alg":"RS256","n":"{n}","e":"{e}"}}]}}"#
        )
    }

    /// Bind 127.0.0.1:0 and serve `body` (as application/json) to every
    /// request that arrives until the test completes. Returns the
    /// `http://host:port/jwks.json` URL. The server task is detached — it
    /// dies when the test runtime tears down.
    async fn start_jwks_server(body: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/jwks.json");
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let body = body.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let _ = stream.read(&mut buf).await;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });
        url
    }

    fn sign_with_test_key(signing_input: &[u8]) -> Vec<u8> {
        let signer = SigningKey::<Sha256>::new(keypair().clone());
        let mut rng = rand::thread_rng();
        signer.sign_with_rng(&mut rng, signing_input).to_bytes().to_vec()
    }

    fn b64(s: &str) -> String {
        URL_SAFE_NO_PAD.encode(s.as_bytes())
    }

    /// Build a JWT from header JSON + claims JSON, signing with our test key.
    fn make_signed_jwt(header_json: &str, claims_json: &str) -> String {
        let signing_input = format!("{}.{}", b64(header_json), b64(claims_json));
        let sig = sign_with_test_key(signing_input.as_bytes());
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(sig))
    }

    /// Like `make_signed_jwt` but lets the caller substitute the signature
    /// segment — for tampered-signature and alg-confusion tests.
    fn make_jwt_with_signature(header_json: &str, claims_json: &str, raw_sig: &[u8]) -> String {
        format!(
            "{}.{}.{}",
            b64(header_json),
            b64(claims_json),
            URL_SAFE_NO_PAD.encode(raw_sig)
        )
    }

    fn now_unix() -> i64 {
        OffsetDateTime::now_utc().unix_timestamp()
    }

    fn good_claims(iss: &str, aud: &str, exp: i64) -> String {
        format!(
            r#"{{"iss":"{iss}","sub":"sub_test","aud":"{aud}","exp":{exp},"email":"u@example.test"}}"#
        )
    }

    fn good_header() -> String {
        format!(r#"{{"alg":"RS256","typ":"{ID_TOKEN_TYP}","kid":"{KID}"}}"#)
    }

    /// A production-issuer config whose JWKS fetch is pointed at the in-process
    /// server. The issuer (`ISS`) has no `/test/{tenant}` suffix, so its
    /// `environment()` is `Production`.
    fn config_pointing_at(jwks_uri: String) -> ChirpAuthConfig {
        ChirpAuthConfig::new(ISS, AUD).with_jwks_uri(jwks_uri)
    }

    // -------------------- happy path --------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_a_well_formed_signed_token_as_human() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let identity = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify");
        assert_eq!(identity.environment, Environment::Prod);
        match identity.identity {
            ChirpVerifiedIdentity::Human { sub, .. } => {
                assert_eq!(sub, "sub_test");
            }
            _ => panic!("expected Human"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_a_resource_access_token_and_exposes_scopes() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS256","typ":"{ACCESS_TOKEN_TYP}","kid":"{KID}"}}"#);
        let exp = now_unix() + 3600;
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"sub_test","aud":"drive","azp":"app_gotta","scope":"drive.deposit","exp":{exp}}}"#
        );
        let token = make_signed_jwt(&header, &claims);
        // The resource server's audience allowlist is the RESOURCE it serves.
        let config = ChirpAuthConfig::new(ISS, "drive").with_jwks_uri(jwks);
        let verified = verify_chirp_resource_access_token(&reqwest::Client::new(), &config, &token)
            .await
            .expect("verify resource access token");
        assert_eq!(verified.environment, Environment::Prod);
        assert_eq!(verified.sub, "sub_test");
        assert_eq!(verified.resource, "drive");
        assert!(verified.has_scope("drive.deposit"));
        assert!(!verified.has_scope("drive.admin"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resource_verifier_rejects_an_id_token() {
        // An ID token (chirp-id+jwt) must never be honored as resource authority:
        // the typ gate rejects it before any claim is trusted.
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, "drive", now_unix() + 3600));
        let config = ChirpAuthConfig::new(ISS, "drive").with_jwks_uri(jwks);
        let err = verify_chirp_resource_access_token(&reqwest::Client::new(), &config, &token)
            .await
            .unwrap_err();
        assert!(matches!(err, ChirpAuthError::InvalidTokenType), "got {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn resource_verifier_rejects_wrong_resource_audience() {
        // A token audience'd to a resource this server does not serve is rejected.
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS256","typ":"{ACCESS_TOKEN_TYP}","kid":"{KID}"}}"#);
        let exp = now_unix() + 3600;
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"sub_test","aud":"calendar","scope":"drive.deposit","exp":{exp}}}"#
        );
        let token = make_signed_jwt(&header, &claims);
        let config = ChirpAuthConfig::new(ISS, "drive").with_jwks_uri(jwks);
        let err = verify_chirp_resource_access_token(&reqwest::Client::new(), &config, &token)
            .await
            .unwrap_err();
        assert!(matches!(err, ChirpAuthError::ClaimMismatch), "got {err:?}");
    }

    // -------------------- adversarial paths --------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_token_with_tampered_signature() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        // Corrupt the FIRST char of the signature segment. The first base64
        // char maps to a full leading byte, so flipping it always changes the
        // signature bytes (→ SignatureInvalid) while staying canonical base64.
        // Flipping the LAST char instead can land on non-canonical trailing
        // bits, which the decoder rejects as MalformedToken — and since the
        // signature varies per run (claims carry a live exp), that made the
        // expected-error assertion flaky.
        let (head, sig) = token.rsplit_once('.').expect("jwt has signature segment");
        let mut sig_chars: Vec<char> = sig.chars().collect();
        sig_chars[0] = if sig_chars[0] == 'A' { 'B' } else { 'A' };
        let tampered: String = format!("{head}.{}", sig_chars.into_iter().collect::<String>());
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &tampered,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::SignatureInvalid), "got {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_expired_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() - 1));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::Expired), "got {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_wrong_audience() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(
            &good_header(),
            &good_claims(ISS, "cs_test_someone_else", now_unix() + 3600),
        );
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::ClaimMismatch), "got {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_wrong_issuer() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(
            &good_header(),
            &good_claims("https://attacker.test", AUD, now_unix() + 3600),
        );
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::ClaimMismatch), "got {err:?}");
    }

    /// Classic issuer-suffix trick: a token whose `iss` ends with the
    /// expected issuer string but is actually a different domain. Exact
    /// equality (not `ends_with`) must reject.
    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_issuer_suffix_attack() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        // Note: not `https://signin.test.example.attacker.test` (suffix
        // append) because the expected ISS doesn't have a trailing slash; the
        // attack the test demonstrates is the symmetric "iss prepended" form
        // that a naive `.contains(expected)` check would let through.
        let attacker_iss = format!("https://attacker.test/path/{ISS}");
        let token = make_signed_jwt(
            &good_header(),
            &good_claims(&attacker_iss, AUD, now_unix() + 3600),
        );
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::ClaimMismatch), "got {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_alg_none_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        // alg=none with empty signature is the classic "none" attack.
        // Our verifier rejects this at jwt_parts (empty signature segment)
        // OR at the alg pin if non-empty. Either rejection is acceptable;
        // assert one of the two well-defined errors.
        let header = format!(r#"{{"alg":"none","typ":"JWT","kid":"{KID}"}}"#);
        let claims = good_claims(ISS, AUD, now_unix() + 3600);
        let token = format!("{}.{}.", b64(&header), b64(&claims));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, ChirpAuthError::MalformedToken | ChirpAuthError::UnsupportedAlgorithm),
            "got {err:?}",
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_explicit_other_algorithm() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS512","typ":"JWT","kid":"{KID}"}}"#);
        let claims = good_claims(ISS, AUD, now_unix() + 3600);
        // Any garbage in the signature slot — verifier rejects on alg pin
        // before touching it.
        let token = make_jwt_with_signature(&header, &claims, &[0u8; 32]);
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::UnsupportedAlgorithm), "got {err:?}");
    }

    /// Header-level class isolation: a token with the key-binding `typ` (a
    /// validly chirp-signed cert) presented to the ID-token verifier is rejected
    /// on `typ` — never honored as an ID token.
    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_keybind_typ_on_id_token_path() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS256","typ":"{KEYBIND_TYP}","kid":"{KID}"}}"#);
        let token = make_signed_jwt(&header, &good_claims(ISS, AUD, now_unix() + 3600));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::InvalidTokenType), "got {err:?}");
    }

    /// A pre-hardening token with the old `typ: "JWT"` (no class marker) is
    /// rejected — there is no legacy/absent acceptance.
    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_legacy_jwt_typ_on_id_token_path() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS256","typ":"JWT","kid":"{KID}"}}"#);
        let token = make_signed_jwt(&header, &good_claims(ISS, AUD, now_unix() + 3600));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::InvalidTokenType), "got {err:?}");
    }

    /// `verify_rs256_jws` rejects a token whose `typ` differs from the caller's
    /// `expected_typ` — an ID token cannot be verified through the keybind path.
    #[tokio::test(flavor = "multi_thread")]
    async fn verify_rs256_jws_enforces_expected_typ() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = config_pointing_at(jwks);
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        // good_header() carries ID_TOKEN_TYP; asking for KEYBIND_TYP must fail.
        let err = verify_rs256_jws(&reqwest::Client::new(), &config, &token, false, KEYBIND_TYP)
            .await
            .unwrap_err();
        assert!(matches!(err, ChirpAuthError::InvalidTokenType), "got {err:?}");
    }

    /// Algorithm-confusion / kid-confusion: token claims an RS256 alg with
    /// a kid the JWKS doesn't advertise. Must fail at key lookup, not
    /// silently accept against some other key.
    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_unknown_kid() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let header = format!(r#"{{"alg":"RS256","typ":"{ID_TOKEN_TYP}","kid":"never-issued"}}"#);
        let claims = good_claims(ISS, AUD, now_unix() + 3600);
        let token = make_jwt_with_signature(&header, &claims, &[0u8; 256]);
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::NoMatchingKey), "got {err:?}");
    }

    /// Machine tokens are rejected by default. Every Drive/Pigeon/Social-
    /// Graph code path that uses the default `VerifyOptions` relies on this.
    /// Confirm the verifier does not silently downgrade a machine token to
    /// a Human identity (and that the rejection happens AFTER signature
    /// validation, so the test mints a real signed machine token).
    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_machine_token_when_accept_machine_false() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let exp = now_unix() + 3600;
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"agent_test","aud":"{AUD}","exp":{exp},"act":"machine","owner_sub":"sub_owner"}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::MachineNotAllowed), "got {err:?}");
    }

    // -------------------- audience allowlist --------------------

    /// A config carrying several accepted audiences must accept a token whose
    /// single `aud` is any one of them — in one pass, no per-audience loop.
    #[tokio::test(flavor = "multi_thread")]
    async fn allowlist_accepts_any_listed_audience() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = ChirpAuthConfig::with_audiences(
            ISS,
            ["cs_one".to_owned(), AUD.to_owned(), "cs_three".to_owned()],
        )
        .with_jwks_uri(jwks);
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify");
        assert!(matches!(verified.identity, ChirpVerifiedIdentity::Human { .. }));
    }

    /// A token whose `aud` is NOT in the allowlist is rejected.
    #[tokio::test(flavor = "multi_thread")]
    async fn allowlist_rejects_unlisted_audience() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = ChirpAuthConfig::with_audiences(
            ISS,
            ["cs_one".to_owned(), "cs_two".to_owned()],
        )
        .with_jwks_uri(jwks);
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::ClaimMismatch), "got {err:?}");
    }

    /// A multi-valued `aud` array is accepted when any one value is allowed.
    #[tokio::test(flavor = "multi_thread")]
    async fn allowlist_accepts_multivalued_aud_intersection() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = ChirpAuthConfig::new(ISS, AUD).with_jwks_uri(jwks);
        let exp = now_unix() + 3600;
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"sub_test","aud":["cs_other","{AUD}"],"exp":{exp},"email":"u@example.test"}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify");
        assert!(matches!(verified.identity, ChirpVerifiedIdentity::Human { .. }));
    }

    // -------------------- verify_from_headers --------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn verify_from_headers_extracts_and_verifies() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        let identity = verify_from_headers(
            &reqwest::Client::new(),
            &headers,
            &config_pointing_at(jwks),
            VerifyOptions::default(),
        )
        .await
        .expect("verify");
        assert!(matches!(identity, ChirpVerifiedIdentity::Human { .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn verify_from_headers_missing_bearer() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let headers = http::HeaderMap::new();
        let err = verify_from_headers(
            &reqwest::Client::new(),
            &headers,
            &config_pointing_at(jwks),
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::MissingBearer), "got {err:?}");
    }

    // -------------------- verify_rs256_jws (low-level) --------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn verify_rs256_jws_returns_claims_and_raw() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = config_pointing_at(jwks);
        let exp = now_unix() + 3600;
        // A key-binding-cert-shaped payload: standard claims plus an extra
        // field the high-level verifier doesn't model.
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"sub_test","aud":"{AUD}","exp":{exp},"bound_pubkey":"ed25519:abc"}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let verified = verify_rs256_jws(&reqwest::Client::new(), &config, &token, true, ID_TOKEN_TYP)
            .await
            .expect("verify");
        assert_eq!(verified.sub, "sub_test");
        assert_eq!(verified.aud, vec![AUD.to_owned()]);
        assert_eq!(
            verified.raw.get("bound_pubkey").and_then(|v| v.as_str()),
            Some("ed25519:abc")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn verify_rs256_jws_skips_aud_when_not_validating() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let config = config_pointing_at(jwks);
        let exp = now_unix() + 3600;
        // aud not in allowlist — accepted because validate_aud = false.
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"sub_test","aud":"cs_unrelated","exp":{exp}}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let verified = verify_rs256_jws(&reqwest::Client::new(), &config, &token, false, ID_TOKEN_TYP)
            .await
            .expect("verify without aud validation");
        assert_eq!(verified.sub, "sub_test");
    }

    /// Machine tokens are rejected by default. (kept as a paired inverse below)
    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_machine_token_when_accept_machine_true() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let exp = now_unix() + 3600;
        // Audience rule: a machine token's `aud` is its OWN client_id, NOT this
        // RP's audience. It is presented to a service whose accepted-audience set
        // does not contain it, and must STILL verify (audience-free bearer
        // assertion). Here `aud` is a foreign client id, distinct from `AUD`.
        let foreign_client = "cs_live_some_other_app";
        let claims = format!(
            r#"{{"iss":"{ISS}","sub":"agent_test","aud":"{foreign_client}","exp":{exp},"act":"machine","owner_sub":"sub_owner"}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let identity = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::accept_machine_clients([foreign_client.to_owned()]),
        )
        .await
        .expect("verify machine (human audience not enforced; client_id allowlisted)");
        assert_eq!(identity.environment, Environment::Prod);
        match identity.identity {
            ChirpVerifiedIdentity::Machine { sub, owner_sub, client_id } => {
                assert_eq!(sub, "agent_test");
                assert_eq!(owner_sub, "sub_owner");
                assert_eq!(client_id, foreign_client);
            }
            _ => panic!("expected Machine"),
        }
    }

    // -------------------- accepted-machine-audiences gate --------------
    //
    // The machine `client_id` membership check now lives in the library (it
    // used to be re-implemented by Drive/Granite/Social Graph). Three cases:
    // an allowed client passes, a disallowed client is rejected, and an empty
    // accepted set fails closed.

    fn machine_token_with_client(client_id: &str) -> String {
        let exp = now_unix() + 3600;
        format!(
            r#"{{"iss":"{ISS}","sub":"agent_test","aud":"{client_id}","exp":{exp},"act":"machine","owner_sub":"sub_owner"}}"#
        )
    }

    /// A machine token whose `client_id` is in the accepted set passes.
    #[tokio::test(flavor = "multi_thread")]
    async fn machine_audience_allowed_client_passes() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let client_id = "cs_live_cg";
        let token = make_signed_jwt(&good_header(), &machine_token_with_client(client_id));
        let identity = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::accept_machine_clients([client_id.to_owned()]),
        )
        .await
        .expect("allowlisted machine client_id accepted");
        match identity.identity {
            ChirpVerifiedIdentity::Machine { client_id: got, .. } => assert_eq!(got, client_id),
            _ => panic!("expected Machine"),
        }
    }

    /// With `accept_any_machine_client`, a machine token from an UNLISTED client
    /// (empty allowlist) is accepted — the grant-scoped opt-in for RPs like Drive
    /// that authorize per-operation via a grant, not by `client_id`.
    #[tokio::test(flavor = "multi_thread")]
    async fn accept_any_machine_client_bypasses_the_allowlist() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &machine_token_with_client("cs_live_unlisted"));
        let opts = VerifyOptions {
            accept_machine: true,
            accept_any_machine_client: true,
            ..Default::default() // accepted_machine_audiences intentionally EMPTY
        };
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            opts,
        )
        .await
        .expect("any machine client accepted under the grant-scoped opt-in");
        match verified.identity {
            ChirpVerifiedIdentity::Machine { client_id, .. } => {
                assert_eq!(client_id, "cs_live_unlisted")
            }
            _ => panic!("expected Machine"),
        }
    }

    /// A machine token whose `client_id` is NOT in the accepted set is rejected
    /// with [`ChirpAuthError::MachineAudienceNotAccepted`] — distinct from
    /// `MachineNotAllowed` (machine tokens ARE accepted here, just not this one).
    #[tokio::test(flavor = "multi_thread")]
    async fn machine_audience_disallowed_client_rejected() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &machine_token_with_client("cs_live_intruder"));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::accept_machine_clients(["cs_live_cg".to_owned()]),
        )
        .await
        .expect_err("non-allowlisted machine client_id rejected");
        assert!(matches!(err, ChirpAuthError::MachineAudienceNotAccepted), "got {err:?}");
    }

    /// `accept_machine: true` with an EMPTY accepted set fails closed: no machine
    /// token gets through, rather than silently trusting every agent.
    #[tokio::test(flavor = "multi_thread")]
    async fn machine_audience_empty_set_rejects() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &machine_token_with_client("cs_live_cg"));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions { accept_machine: true, ..Default::default() },
        )
        .await
        .expect_err("empty accepted-machine-audiences set rejects all machine tokens");
        assert!(matches!(err, ChirpAuthError::MachineAudienceNotAccepted), "got {err:?}");
    }

    // -------------------- consumer-profile contract tests --------------
    //
    // Three downstream services consume this crate and each calls
    // verify_chirp_id_token with a different VerifyOptions config:
    //
    //   Drive        → accept machine, gated to its CG client_id(s)
    //   Pigeon       → accept machine, gated to its CG client_id(s)
    //   Social Graph → human only (accept_machine: false)
    // There is no email axis — email is never in a token or the verified identity.
    // The accepted-machine-audiences gate now lives in the library; the profile
    // helpers below name `AUD` as the one machine client these services accept
    // (the machine fixtures mint with `aud == AUD`).

    fn drive_profile() -> VerifyOptions {
        VerifyOptions::accept_machine_clients([AUD.to_owned()])
    }
    fn pigeon_profile() -> VerifyOptions {
        VerifyOptions::accept_machine_clients([AUD.to_owned()])
    }
    fn social_graph_profile() -> VerifyOptions {
        VerifyOptions { accept_machine: false, ..Default::default() }
    }

    fn human_claims_with_email(iss: &str, aud: &str, exp: i64) -> String {
        format!(
            r#"{{"iss":"{iss}","sub":"sub_user","aud":"{aud}","exp":{exp},"email":"u@example.test"}}"#
        )
    }
    fn human_claims_without_email(iss: &str, aud: &str, exp: i64) -> String {
        format!(
            r#"{{"iss":"{iss}","sub":"sub_user","aud":"{aud}","exp":{exp}}}"#
        )
    }
    fn machine_claims(iss: &str, aud: &str, exp: i64) -> String {
        format!(
            r#"{{"iss":"{iss}","sub":"agent_x","aud":"{aud}","exp":{exp},"act":"machine","owner_sub":"sub_owner"}}"#
        )
    }

    async fn run(profile: VerifyOptions, claims: &str) -> Result<ChirpVerifiedIdentity, ChirpAuthError> {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), claims);
        verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            profile,
        )
        .await
        .map(|verified| verified.identity)
    }

    // -- Drive profile -------------------------------------------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn drive_profile_accepts_human_with_email() {
        let claims = human_claims_with_email(ISS, AUD, now_unix() + 3600);
        let id = run(drive_profile(), &claims).await.expect("accept");
        assert!(matches!(id, ChirpVerifiedIdentity::Human { .. }));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn drive_profile_accepts_human_without_email() {
        let claims = human_claims_without_email(ISS, AUD, now_unix() + 3600);
        let id = run(drive_profile(), &claims).await.expect("accept");
        assert!(matches!(id, ChirpVerifiedIdentity::Human { .. }));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn drive_profile_accepts_machine_token() {
        let claims = machine_claims(ISS, AUD, now_unix() + 3600);
        let id = run(drive_profile(), &claims).await.expect("accept");
        assert!(matches!(id, ChirpVerifiedIdentity::Machine { .. }));
    }

    // -- Pigeon profile (currently identical to Drive's) ---------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn pigeon_profile_matches_drive_profile() {
        assert_eq!(
            drive_profile().accept_machine,
            pigeon_profile().accept_machine,
            "Drive and Pigeon profiles diverged. Update this test if intentional.",
        );
    }

    // -- Social Graph profile ------------------------------------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn social_graph_profile_accepts_human() {
        // Email is never in the identity; Social Graph accepts any human token.
        let claims = human_claims_without_email(ISS, AUD, now_unix() + 3600);
        let id = run(social_graph_profile(), &claims).await.expect("accept");
        assert!(matches!(id, ChirpVerifiedIdentity::Human { .. }));
    }
    #[tokio::test(flavor = "multi_thread")]
    async fn social_graph_profile_rejects_machine_token() {
        let claims = machine_claims(ISS, AUD, now_unix() + 3600);
        let err = run(social_graph_profile(), &claims).await.unwrap_err();
        assert!(matches!(err, ChirpAuthError::MachineNotAllowed), "got {err:?}");
    }

    // -------------------- environment enforcement --------------------
    //
    // The environment axis is now enforced. A production-configured RP must
    // reject a `test: true` token; a test-configured RP must reject a
    // non-test token. Both fail with EnvironmentMismatch.

    fn human_test_claims(iss: &str, aud: &str, exp: i64) -> String {
        format!(
            r#"{{"iss":"{iss}","sub":"sub_test","aud":"{aud}","exp":{exp},"email":"u@example.test","test":true}}"#
        )
    }

    /// A test issuer config (`…/test/{tenant}`). The JWKS is the same
    /// in-process server (in production the test keyset is disjoint, but for
    /// the verify-path test what matters is the issuer string the config and
    /// token agree on).
    fn test_issuer_config(jwks_uri: String) -> (ChirpAuthConfig, String) {
        let test_iss = format!("{ISS}/test/acme");
        let config = ChirpAuthConfig::new(&test_iss, AUD).with_jwks_uri(jwks_uri);
        (config, test_iss)
    }

    /// Production RP + production (non-test) token → accepted.
    #[tokio::test(flavor = "multi_thread")]
    async fn prod_rp_accepts_prod_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify");
        assert_eq!(verified.environment, Environment::Prod);
    }

    /// Production RP + test token → rejected with EnvironmentMismatch. This is
    /// the core enforcement: a production relying party can never honor a token
    /// minted by a test deployment, even with a valid signature/iss/aud/exp.
    #[tokio::test(flavor = "multi_thread")]
    async fn prod_rp_rejects_test_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &human_test_claims(ISS, AUD, now_unix() + 3600));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::EnvironmentMismatch), "got {err:?}");
    }

    /// Test RP + test token → accepted, and reported as Environment::Test.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_rp_accepts_test_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let (config, test_iss) = test_issuer_config(jwks);
        let token =
            make_signed_jwt(&good_header(), &human_test_claims(&test_iss, AUD, now_unix() + 3600));
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify test token under test RP");
        assert_eq!(verified.environment, Environment::Test { tenant: "acme".into() });
        // A non-machine token under a test issuer is a `Test` principal, NOT a
        // `Human` — the class follows the verified keyset's environment so a
        // test identity is structurally distinct from a production human.
        match verified.identity {
            ChirpVerifiedIdentity::Test { sub } => assert_eq!(sub, "sub_test"),
            other => panic!("expected Test, got {other:?}"),
        }
    }

    /// A TEST-KEYSET MACHINE token verifies as `Machine` + `Environment::Test`.
    /// Its test-ness is purely the issuer/keyset — it carries NO `test: true`
    /// claim (a real chirp-minted machine token never can: `act: "machine"` +
    /// `test: true` is unrepresentable at mint, INV-09), and its `test:`-
    /// namespaced subs pass verify unchanged (the Machine arm does not constrain
    /// the sub form). This is the shape the Warden agent-under-grant E2E rides.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_rp_machine_token_is_machine_not_test() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let (config, test_iss) = test_issuer_config(jwks);
        let exp = now_unix() + 3600;
        let foreign_client = "cs_live_agent";
        // A real test-machine: act:"machine", test:-namespaced subs, NO test claim.
        let claims = format!(
            r#"{{"iss":"{test_iss}","sub":"test:agent_probe","aud":"{foreign_client}","exp":{exp},"act":"machine","owner_sub":"test:sub_owner"}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let verified = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::accept_machine_clients([foreign_client.to_owned()]),
        )
        .await
        .expect("verify test-keyset machine token");
        assert_eq!(verified.environment, Environment::Test { tenant: "acme".into() });
        match verified.identity {
            ChirpVerifiedIdentity::Machine { sub, owner_sub, .. } => {
                assert_eq!(sub, "test:agent_probe");
                assert_eq!(owner_sub, "test:sub_owner");
            }
            other => panic!("expected Machine, got {other:?}"),
        }
    }

    /// A machine token that smuggles `test: true` is rejected as malformed.
    /// `act: "machine"` + `test: true` is unrepresentable at mint (INV-09), so
    /// its presence means a hand-forged token — never accept it as either a
    /// production or a test machine.
    #[tokio::test(flavor = "multi_thread")]
    async fn machine_token_with_test_claim_is_malformed() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let (config, test_iss) = test_issuer_config(jwks);
        let exp = now_unix() + 3600;
        let foreign_client = "cs_live_agent";
        let claims = format!(
            r#"{{"iss":"{test_iss}","sub":"test:agent_probe","aud":"{foreign_client}","exp":{exp},"act":"machine","owner_sub":"test:sub_owner","test":true}}"#
        );
        let token = make_signed_jwt(&good_header(), &claims);
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::accept_machine_clients([foreign_client.to_owned()]),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::MalformedMachineToken), "got {err:?}");
    }

    /// Test RP + non-test token → rejected. The symmetric direction: a test
    /// deployment must not honor a production-shaped token.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_rp_rejects_non_test_token() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let (config, test_iss) = test_issuer_config(jwks);
        // Non-test claims (no `test: true`) but with the test issuer.
        let token = make_signed_jwt(&good_header(), &good_claims(&test_iss, AUD, now_unix() + 3600));
        let err = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config,
            &token,
            VerifyOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ChirpAuthError::EnvironmentMismatch), "got {err:?}");
    }

    /// A non-test token must still be accepted under the default options by a
    /// production RP — the environment gate only fires on disagreement.
    #[tokio::test(flavor = "multi_thread")]
    async fn accepts_non_test_token_by_default() {
        let jwks = start_jwks_server(jwks_body_with_test_key()).await;
        let token = make_signed_jwt(&good_header(), &good_claims(ISS, AUD, now_unix() + 3600));
        let identity = verify_chirp_id_token(
            &reqwest::Client::new(),
            &config_pointing_at(jwks),
            &token,
            VerifyOptions::default(),
        )
        .await
        .expect("verify non-test token");
        assert!(matches!(identity.identity, ChirpVerifiedIdentity::Human { .. }));
    }

    #[test]
    fn environment_from_issuer_classifies_provenance() {
        let test = |tenant: &str| Environment::Test {
            tenant: tenant.to_owned(),
        };
        let prod = Environment::Prod;
        assert_eq!(environment_from_issuer("https://signin.chirpauth.com"), prod);
        assert_eq!(environment_from_issuer("https://signin.chirpauth.com/"), prod);
        // The tenant SURVIVES classification — this is the whole point of
        // riding capability::Environment rather than a tenant-less fork.
        assert_eq!(
            environment_from_issuer("https://signin.chirpauth.com/test/acme"),
            test("acme")
        );
        assert_eq!(
            environment_from_issuer("https://signin.chirpauth.com/test/acme/"),
            test("acme")
        );
        // Distinct tenants are distinct environments — `Environment` derives Eq
        // per-tenant, so this is what stops one test tenant being mistaken for
        // another downstream.
        assert_ne!(
            environment_from_issuer("https://signin.chirpauth.com/test/acme"),
            environment_from_issuer("https://signin.chirpauth.com/test/other")
        );
        // multi-segment after /test/ is not a valid single-tenant test issuer
        assert_eq!(
            environment_from_issuer("https://signin.chirpauth.com/test/a/b"),
            prod
        );
        // empty tenant, and a host merely named "test", are production
        assert_eq!(
            environment_from_issuer("https://signin.chirpauth.com/test/"),
            prod
        );
        assert_eq!(environment_from_issuer("https://test.example.com"), prod);
    }

    // -------------------- environment enforcement, quantified --------------
    //
    // Property P1 — environment/keyset soundness on the verifier side. The four
    // hand-written direction tests above spot-check the prod/test × prod/test
    // matrix with one issuer each. This quantifies the same enforcement over an
    // arbitrary tenant id AND both axes at once: with signature, iss, aud, and
    // exp all valid in every case (so the environment axis is isolated), the
    // verifier accepts IFF the issuer-derived environment matches the token's
    // `test` provenance. A test token can never be honored by a production RP,
    // and vice-versa, no matter how the tenant is spelled.
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

        #[test]
        fn env_enforced_accept_iff_environments_match(
            tenant in "[a-z0-9]{1,16}",
            use_test_issuer in any::<bool>(),
            token_is_test in any::<bool>(),
        ) {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(async {
                let jwks = start_jwks_server(jwks_body_with_test_key()).await;
                let issuer = if use_test_issuer {
                    format!("{ISS}/test/{tenant}")
                } else {
                    ISS.to_string()
                };
                let config = ChirpAuthConfig::new(&issuer, AUD).with_jwks_uri(jwks);
                // The issuer decides the environment (and, when test, the
                // tenant); the token's `test` claim carries only a bool. The
                // verifier accepts iff those two agree on test-ness.
                let expected_env = if use_test_issuer {
                    Environment::Test {
                        tenant: tenant.clone(),
                    }
                } else {
                    Environment::Prod
                };

                let test_field = if token_is_test { r#","test":true"# } else { "" };
                let claims = format!(
                    r#"{{"iss":"{issuer}","sub":"sub_p","aud":"{AUD}","exp":{}{test_field}}}"#,
                    now_unix() + 3600,
                );
                let token = make_signed_jwt(&good_header(), &claims);
                let result = verify_chirp_id_token(
                    &reqwest::Client::new(),
                    &config,
                    &token,
                    VerifyOptions::default(),
                )
                .await;

                let should_accept = use_test_issuer == token_is_test;
                prop_assert_eq!(
                    result.is_ok(),
                    should_accept,
                    "issuer={} token_is_test={}",
                    issuer,
                    token_is_test
                );
                if let Ok(verified) = result {
                    prop_assert_eq!(verified.environment, expected_env);
                }
                Ok(())
            })?;
        }
    }
}
