--[[
Cross-reference Lua filter for Pandoc

Converts custom cross-reference syntax to LaTeX labels and refs.

Anchor definitions (placed after equations, definitions, etc.):
    {#EQ-1.1}  → \label{EQ-1.1}
    {#DEF-1.2} → \label{DEF-1.2}
    {#THM-1.3} → \label{THM-1.3}

Inline references:
    #EQ-1.1    → \eqref{EQ-1.1}   (equation refs use eqref)
    #DEF-1.2   → \ref{DEF-1.2}    (others use ref)
    #THM-1.3   → \ref{THM-1.3}

Bracket references:
    [EQ-1.1]   → \eqref{EQ-1.1}
    [DEF-1.2]  → \ref{DEF-1.2}

Supported prefixes:
    EQ-   : Equations (uses \eqref for proper formatting)
    DEF-  : Definitions
    THM-  : Theorems
    REM-  : Remarks
    ASM-  : Assumptions
    WDEF- : Working Definitions

Usage:
    pandoc input.md --lua-filter=crossrefs.lua -o output.tex
--]]

-- Prefixes that should use \eqref (for equations)
local EQUATION_PREFIXES = {
    ["EQ"] = true,
}

-- All valid prefixes for cross-references
local VALID_PREFIXES = {
    ["EQ"] = true,
    ["DEF"] = true,
    ["THM"] = true,
    ["REM"] = true,
    ["ASM"] = true,
    ["WDEF"] = true,
}

--- Check if a reference ID starts with an equation prefix
-- @param id string The reference ID (e.g., "EQ-1.1")
-- @return boolean True if this is an equation reference
local function is_equation_ref(id)
    local prefix = id:match("^([A-Z]+)%-")
    return prefix and EQUATION_PREFIXES[prefix]
end

--- Check if a string is a valid cross-reference ID
-- @param id string The potential reference ID
-- @return boolean True if this is a valid reference
local function is_valid_ref(id)
    local prefix = id:match("^([A-Z]+)%-")
    return prefix and VALID_PREFIXES[prefix]
end

--- Generate LaTeX reference command for an ID
-- @param id string The reference ID (e.g., "EQ-1.1")
-- @return string LaTeX command (\eqref{} or \ref{})
local function make_ref_command(id)
    if is_equation_ref(id) then
        return "\\eqref{" .. id .. "}"
    else
        return "\\ref{" .. id .. "}"
    end
end

--- Process Span elements (handles {#ID} anchor syntax)
-- Pandoc parses {#ID} as a Span with that identifier
function Span(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Check if this span has an identifier that matches our pattern
    if el.identifier and el.identifier ~= "" then
        if is_valid_ref(el.identifier) then
            -- This is an anchor definition - emit \label{}
            -- Keep any content inside the span, add label after
            local label = pandoc.RawInline("latex", "\\label{" .. el.identifier .. "}")

            if #el.content == 0 then
                -- Empty span, just return the label
                return label
            else
                -- Span has content, append label to content
                local result = pandoc.List(el.content)
                result:insert(label)
                return result
            end
        end
    end

    return nil
end

--- Process Str elements (handles #ID and [ID] reference syntax)
function Str(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    local text = el.text

    -- Pattern 1: Hash reference #TYPE-X.Y or #TYPE-X.Y.Z or #TYPE-X.Y-suffix
    -- Must end with alphanumeric (not a dot - that's sentence punctuation)
    local hash_ref = text:match("^#([A-Z]+%-[%w%.%-]*%w)(.?)$")
    if hash_ref and is_valid_ref(hash_ref) then
        local trailing = text:match("^#[A-Z]+%-[%w%.%-]*%w(.)$") or ""
        if trailing == "" then
            return pandoc.RawInline("latex", make_ref_command(hash_ref))
        else
            -- There's trailing punctuation (like a period) - keep it
            return {
                pandoc.RawInline("latex", make_ref_command(hash_ref)),
                pandoc.Str(trailing)
            }
        end
    end

    -- Also handle case where ref is followed by punctuation in same token
    local ref_with_punct, punct = text:match("^#([A-Z]+%-[%w%.%-]*%w)([%.,;:!?])$")
    if ref_with_punct and is_valid_ref(ref_with_punct) then
        return {
            pandoc.RawInline("latex", make_ref_command(ref_with_punct)),
            pandoc.Str(punct)
        }
    end

    -- Pattern 2: Bracket reference [TYPE-X.Y] (as a single Str token)
    -- Note: Pandoc usually breaks [text] into Link, but sometimes keeps as Str
    local bracket_ref = text:match("^%[([A-Z]+%-[%w%.%-]*%w)%]$")
    if bracket_ref and is_valid_ref(bracket_ref) then
        return pandoc.RawInline("latex", make_ref_command(bracket_ref))
    end

    return nil
end

--- Process Link elements (handles [ID] that Pandoc parses as links)
-- When Pandoc sees [EQ-1.1] it may parse it as a link with empty target
function Link(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Check if this is a reference-style "link" with no URL
    -- [EQ-1.1] becomes a Link with target "" or "#EQ-1.1"
    if el.target == "" or el.target:match("^#") then
        -- Extract the link text
        local link_text = pandoc.utils.stringify(el.content)

        -- Check if it matches our reference pattern
        if is_valid_ref(link_text) then
            return pandoc.RawInline("latex", make_ref_command(link_text))
        end

        -- Also check target (for [text](#EQ-1.1) style)
        if el.target:match("^#") then
            local target_ref = el.target:sub(2)  -- Remove leading #
            if is_valid_ref(target_ref) then
                return pandoc.RawInline("latex", make_ref_command(target_ref))
            end
        end
    end

    return nil
end

--- Process Div elements to find anchors in header attributes
-- Headers like "## Section {#DEF-1.1}" get the ID in the Div/Header
function Header(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Check if header has our reference pattern as ID
    if el.identifier and is_valid_ref(el.identifier) then
        -- Add label after the header content
        local label = pandoc.RawInline("latex", "\\label{" .. el.identifier .. "}")

        -- Create new header with label appended to content
        local new_content = pandoc.List(el.content)
        new_content:insert(label)

        return pandoc.Header(el.level, new_content, pandoc.Attr(el.identifier, el.classes, el.attributes))
    end

    return nil
end

-- Return the filter functions
return {
    { Span = Span },
    { Str = Str },
    { Link = Link },
    { Header = Header },
}
