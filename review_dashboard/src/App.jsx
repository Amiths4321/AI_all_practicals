import { useEffect, useState } from "react";

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [resolvedReviews, setResolvedReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedReview, setSelectedReview] = useState(null);
  const [pendingDecision, setPendingDecision] = useState(null);

  const [reviewerName, setReviewerName] = useState("");
  const [reviewerComment, setReviewerComment] = useState("");

  const [statusFilter, setStatusFilter] = useState("ALL");
  const [searchTerm, setSearchTerm] = useState("");
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [apiStatus, setApiStatus] = useState("CHECKING");

  const loadDashboard = () => {
    setLoading(true);
    setApiStatus("CHECKING");

    fetch("http://127.0.0.1:8000/api/reviews/dashboard")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load dashboard");
        }

        return response.json();
      })
      .then((data) => {
        setDashboard(data);
        setApiStatus("CONNECTED");
        setLastRefreshed(new Date());

        const resolved = [
          ...(data.approved_reviews || []),
          ...(data.rejected_reviews || [])
        ];

        setResolvedReviews(resolved);
        setLoading(false);
        setError("");
      })
      .catch((err) => {
        setApiStatus("DISCONNECTED");
        setError(err.message);
        setLoading(false);
      });
  };

  const checkApiHealth = () => {
    fetch("http://127.0.0.1:8000/api/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("API health check failed");
        }

        return response.json();
      })
      .then((data) => {
        setApiStatus(data.status === "healthy" ? "CONNECTED" : "DEGRADED");
      })
      .catch(() => {
        setApiStatus("DISCONNECTED");
      });
  };

  useEffect(() => {
    loadDashboard();
    checkApiHealth();

    const interval = setInterval(() => {
      loadDashboard();
      checkApiHealth();
    }, 30000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const resolveReview = (reviewId, decision) => {
    if (!reviewerName.trim()) {
      setError("Reviewer name is required.");
      return;
    }

    fetch(
      `http://127.0.0.1:8000/api/reviews/${reviewId}/resolve`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          decision: decision,
          reviewer: reviewerName,
          reviewer_comment: reviewerComment
        })
      }
    )
      .then(async (response) => {
        if (!response.ok) {
          const errorData = await response.json();

          throw new Error(
            errorData.detail || "Failed to resolve review"
          );
        }

        return response.json();
      })
      .then(() => {
        setSelectedReview(null);
        setPendingDecision(null);
        setReviewerName("");
        setReviewerComment("");
        setError("");

        loadDashboard();
      })
      .catch((err) => {
        setError(err.message);
      });
  };

  if (loading) {
    return (
      <div
        style={{
          padding: "40px",
          fontFamily: "Arial, sans-serif"
        }}
      >
        <h2>Loading dashboard...</h2>
      </div>
    );
  }

  if (error && !dashboard) {
    return (
      <div
        style={{
          padding: "40px",
          fontFamily: "Arial, sans-serif"
        }}
      >
        <h2>Dashboard Error</h2>

        <p>{error}</p>

        <button
          onClick={() => {
            setError("");
            loadDashboard();
          }}
          style={{
            padding: "10px 18px",
            cursor: "pointer"
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  const allReviews = [
    ...(dashboard?.pending_reviews || []),
    ...(dashboard?.approved_reviews || []),
    ...(dashboard?.rejected_reviews || [])
  ];

  const filteredReviews = allReviews.filter((review) => {
    const matchesStatus =
      statusFilter === "ALL" ||
      review.status === statusFilter;

    const search = searchTerm.toLowerCase();

    const matchesSearch =
      !search ||
      String(review.review_id || "").toLowerCase().includes(search) ||
      String(review.case_id || "").toLowerCase().includes(search) ||
      String(review.product || "").toLowerCase().includes(search) ||
      String(review.reason || "").toLowerCase().includes(search) ||
      String(review.status || "").toLowerCase().includes(search);

    return matchesStatus && matchesSearch;
});

const filteredCount = filteredReviews.length;

  return (
    <div
      style={{
        padding: "30px",
        fontFamily: "Arial, sans-serif"
      }}
    >
      {/* Header */}

      <h1>Banking RAG — Human Review Dashboard</h1>

      <p>
        Review cases requiring human validation.
      </p>

      <p>
        API Status: <strong>{apiStatus}</strong>
      </p>

      {lastRefreshed && (
        <p
          style={{
            color: "#666",
            fontSize: "14px"
          }}
        >
          Last refreshed:{" "}
          {lastRefreshed.toLocaleTimeString()}
        </p>
      )}

      <button
        onClick={loadDashboard}
        disabled={loading}
        style={{
          padding: "10px 18px",
          marginTop: "10px",
          cursor: loading
            ? "not-allowed"
            : "pointer"
        }}
      >
        {loading
          ? "Refreshing..."
          : "Refresh Dashboard"}
      </button>

      <hr />

   

      {/* Dashboard Metrics */}

      <div
        style={{
          display: "flex",
          gap: "20px",
          marginTop: "30px",
          marginBottom: "30px"
        }}
      >
        <div
          onClick={() => setStatusFilter("ALL")}
          style={{
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "8px",
            background: "white",
            cursor: "pointer",
            minWidth: "160px"
          }}
        >
          <h3>Total Reviews</h3>
          <h2>{dashboard.total}</h2>
        </div>

        <div
          onClick={() => setStatusFilter("PENDING")}
          style={{
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "8px",
            background: "white",
            cursor: "pointer",
            minWidth: "160px"
          }}
        >
          <h3>Pending</h3>
          <h2>{dashboard.pending}</h2>
        </div>

        <div
          onClick={() => setStatusFilter("APPROVED")}
          style={{
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "8px",
            background: "white",
            cursor: "pointer",
            minWidth: "160px"
          }}
        >
          <h3>Approved</h3>
          <h2>{dashboard.approved}</h2>
        </div>

        <div
          onClick={() => setStatusFilter("REJECTED")}
          style={{
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "8px",
            background: "white",
            cursor: "pointer",
            minWidth: "160px"
          }}
        >
          <h3>Rejected</h3>
          <h2>{dashboard.rejected}</h2>
        </div>
      </div>

      {/* Pending Alert */}

      {dashboard.pending > 0 && (
        <div
          style={{
            marginTop: "20px",
            padding: "15px",
            border: "1px solid #ccc",
            borderRadius: "8px",
            background: "#fff3cd"
          }}
        >
          <strong>
            Attention: {dashboard.pending} review
            {dashboard.pending === 1
              ? ""
              : "s"} pending.
          </strong>

          <p>
            These cases require human validation.
          </p>

          <button
            onClick={() =>
              setStatusFilter("PENDING")
            }
            style={{
              padding: "9px 16px",
              cursor: "pointer"
            }}
          >
            View Pending Reviews
          </button>
        </div>
      )}

      <hr />

      {/* Filter */}

      <div
        style={{
          marginTop: "30px",
          padding: "15px",
          background: "#f8f9fa",
          border: "1px solid #ddd",
          borderRadius: "8px"
        }}
      >
        <strong>Filter Reviews: </strong>
        <input
          type="text"
          placeholder="Search reviews..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            padding: "9px 12px",
            border: "1px solid #ccc",
            borderRadius: "6px",
            width: "240px",
            marginRight: "10px"
          }}
        />
        <select
          value={statusFilter}
          onChange={(e) =>
            setStatusFilter(e.target.value)
          }
          style={{
            marginLeft: "10px",
            padding: "8px"
          }}
        >
          <option value="ALL">All</option>
          <option value="PENDING">Pending</option>
          <option value="APPROVED">Approved</option>
          <option value="REJECTED">Rejected</option>
        </select>

        <button
          onClick={() => {
            setStatusFilter("ALL");
            setSearchTerm("");
          }}
          style={{
            marginLeft: "10px",
            padding: "8px 14px",
            cursor: "pointer"
          }}
        >
          Clear Filter
        </button>
      </div>

      {/* Review List */}

      <h2>
        {statusFilter === "ALL"
          ? "All Reviews"
          : `${statusFilter
              .charAt(0)
              .toUpperCase()}${statusFilter
              .slice(1)
              .toLowerCase()} Reviews`}
        {" "}({filteredCount})
      </h2>

      {filteredReviews.length === 0 ? (
        <p>
          No{" "}
          {statusFilter === "ALL"
            ? ""
            : statusFilter.toLowerCase() + " "}
          reviews found.
        </p>
      ) : (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            marginTop: "20px"
          }}
        >
          <thead>
            <tr>
              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Review ID
              </th>

              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Product
              </th>

              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Reason
              </th>

              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Query
              </th>

              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Status
              </th>

              <th
                style={{
                  border: "1px solid #ddd",
                  padding: "10px",
                  textAlign: "left"
                }}
              >
                Action
              </th>
            </tr>
          </thead>

          <tbody>
            {filteredReviews.map((review) => (
              <tr
                key={review.review_id}
                onClick={() =>
                  setSelectedReview(review)
                }
                style={{
                  cursor: "pointer",
                  borderBottom: "1px solid #ddd"
                }}
              >
                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.review_id}
                </td>

                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.product}
                </td>

                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.reason}
                </td>

                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.query}
                </td>

                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.status}
                </td>

                <td
                  style={{
                    border: "1px solid #ddd",
                    padding: "10px"
                  }}
                >
                  {review.status === "PENDING" ? (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedReview(review);
                      }}
                    >
                      Review
                    </button>
                  ) : (
                    <span>Resolved</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Review Details */}

      {selectedReview && (
        <div
          style={{
            marginTop: "30px",
            padding: "25px",
            border: "1px solid #ccc",
            borderRadius: "8px",
            background: "#f8f9fa"
          }}
        >
          <h2>Review Details</h2>

          {/* Case Summary */}

          <div
            style={{
              padding: "15px",
              background: "white",
              border: "1px solid #ddd",
              borderRadius: "8px",
              marginBottom: "20px"
            }}
          >
            <h3>Case Summary</h3>

            <p>
              <strong>Review ID:</strong>{" "}
              {selectedReview.review_id}
            </p>

            <p>
              <strong>Product:</strong>{" "}
              {selectedReview.product}
            </p>

            <p>
              <strong>Reason:</strong>{" "}
              {selectedReview.reason}
            </p>

            <p>
              <strong>Query:</strong>{" "}
              {selectedReview.query}
            </p>

            <p>
              <strong>Status:</strong>{" "}
              {selectedReview.status}
            </p>
          </div>

            {selectedReview.status !== "PENDING" && (
              <div
                style={{
                  padding: "15px",
                  background: "white",
                  border: "1px solid #ddd",
                  borderRadius: "8px",
                  marginBottom: "20px"
                }}
              >
                <h3>Resolution Details</h3>

                <p>
                  <strong>Decision:</strong>{" "}
                  {selectedReview.status}
                </p>

                <p>
                  <strong>Reviewer:</strong>{" "}
                  {selectedReview.reviewer || "-"}
                </p>

                <p>
                  <strong>Comment:</strong>{" "}
                  {selectedReview.reviewer_comment || "-"}
                </p>

                <p>
                  <strong>Resolved At:</strong>{" "}
                  {selectedReview.resolved_at || "-"}
                </p>
              </div>
            )}
          {/* Generated Answer */}

          <h3>Generated Answer</h3>

          <div
            style={{
              padding: "15px",
              background: "white",
              border: "1px solid #ddd",
              borderRadius: "8px"
            }}
          >
            {selectedReview.answer ||
              "No answer available."}
          </div>

          {/* Evidence */}

          <h3>Evidence</h3>

          {selectedReview.evidence &&
          selectedReview.evidence.length > 0 ? (
            selectedReview.evidence.map((item) => (
              <div
                key={item.evidence_id}
                style={{
                  marginBottom: "15px",
                  padding: "15px",
                  background: "white",
                  border: "1px solid #ddd",
                  borderRadius: "8px"
                }}
              >
                <h4 style={{ marginTop: "0" }}>
                  Evidence {item.evidence_id}
                </h4>

                <p>
                  <strong>Document:</strong>{" "}
                  {item.document || "-"}
                </p>

                <p>
                  <strong>Section:</strong>{" "}
                  {item.section || "-"}
                </p>

                <p>
                  <strong>Policy Version:</strong>{" "}
                  {item.policy_version || "-"}
                </p>

                <p>
                  <strong>Effective Date:</strong>{" "}
                  {item.effective_date || "-"}
                </p>

                <div
                  style={{
                    marginTop: "10px",
                    padding: "12px",
                    background: "#f8f9fa",
                    borderRadius: "6px"
                  }}
                >
                  <strong>Evidence Text</strong>

                  <p
                    style={{
                      marginBottom: "0"
                    }}
                  >
                    {item.text ||
                      "No evidence text available."}
                  </p>
                </div>
              </div>
            ))
          ) : (
            <p>No evidence available.</p>
          )}

          {/* Human Decision */}

          <div
            style={{
              marginTop: "25px",
              padding: "20px",
              background: "white",
              border: "1px solid #ccc",
              borderRadius: "8px"
            }}
          >
            <h3>Human Decision</h3>

            <p>
              Review the generated answer and
              supporting evidence before making a
              decision.
            </p>

            {selectedReview.status === "PENDING" ? (
              <div>
                <button
                  onClick={() =>
                    setPendingDecision("APPROVED")
                  }
                  style={{
                    padding: "10px 18px",
                    marginRight: "10px",
                    cursor: "pointer"
                  }}
                >
                  Approve Review
                </button>

                <button
                  onClick={() =>
                    setPendingDecision("REJECTED")
                  }
                  style={{
                    padding: "10px 18px",
                    cursor: "pointer"
                  }}
                >
                  Reject Review
                </button>
              </div>
            ) : (
              <p>
                This review has already been resolved.
              </p>
            )}
          </div>

          {/* Confirmation */}

          {pendingDecision && (
            <div
              style={{
                marginTop: "20px",
                padding: "20px",
                border: "2px solid #ccc",
                borderRadius: "8px",
                background: "#fff"
              }}
            >
              <h3>Confirm Decision</h3>

              <p>
                You are about to mark this review as:
              </p>

              <h3>{pendingDecision}</h3>

              <label>
                <strong>Reviewer Name</strong>
              </label>

              <br />

              <input
                type="text"
                value={reviewerName}
                onChange={(e) =>
                  setReviewerName(e.target.value)
                }
                placeholder="Enter reviewer name"
                style={{
                  width: "100%",
                  padding: "10px",
                  marginTop: "8px",
                  marginBottom: "15px",
                  boxSizing: "border-box"
                }}
              />

              <label>
                <strong>Reviewer Comment</strong>
              </label>

              <br />

              <textarea
                value={reviewerComment}
                onChange={(e) =>
                  setReviewerComment(e.target.value)
                }
                placeholder="Enter review comment"
                rows="4"
                style={{
                  width: "100%",
                  padding: "10px",
                  marginTop: "8px",
                  marginBottom: "15px",
                  boxSizing: "border-box"
                }}
              />

              <p>
                Are you sure you want to continue?
              </p>

              <button
                disabled={!reviewerName.trim()}
                onClick={() => {
                  resolveReview(
                    selectedReview.review_id,
                    pendingDecision
                  );
                }}
                style={{
                  padding: "10px 18px",
                  cursor: reviewerName.trim()
                    ? "pointer"
                    : "not-allowed"
                }}
              >
                Confirm {pendingDecision}
              </button>

              {" "}

              <button
                onClick={() =>
                  setPendingDecision(null)
                }
                style={{
                  padding: "10px 18px",
                  cursor: "pointer"
                }}
              >
                Cancel
              </button>
            </div>
          )}

          <button
            onClick={() => {
              setSelectedReview(null);
              setPendingDecision(null);
              setReviewerName("");
              setReviewerComment("");
            }}
            style={{
              marginTop: "20px",
              padding: "10px 18px",
              cursor: "pointer"
            }}
          >
            Close
          </button>
        </div>
      )}

      {/* Resolved Reviews */}

      <hr
        style={{
          marginTop: "40px"
        }}
      />

      <h2>Resolved Reviews</h2>

      {resolvedReviews.length === 0 ? (
        <p>No resolved reviews yet.</p>
      ) : (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            marginTop: "20px"
          }}
        >
          <thead>
            <tr>
              <th>Review ID</th>
              <th>Product</th>
              <th>Reason</th>
              <th>Status</th>
              <th>Reviewer</th>
              <th>Comment</th>
              <th>Resolved At</th>
            </tr>
          </thead>

          <tbody>
            {resolvedReviews.map((review) => (
              <tr key={review.review_id}>
                <td>{review.review_id}</td>
                <td>{review.product}</td>
                <td>{review.reason}</td>
                <td>{review.status}</td>
                <td>{review.reviewer || "-"}</td>
                <td>
                  {review.reviewer_comment || "-"}
                </td>
                <td>{review.resolved_at || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;